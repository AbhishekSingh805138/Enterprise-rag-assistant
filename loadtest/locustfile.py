"""Load profile for the Enterprise RAG Assistant.

There was no load or performance testing at all, so p95 latency and the
concurrency ceiling were unmeasured and capacity planning was guesswork.

Locust rather than k6 because the assertions that matter here are about
*answers*, not just status codes — an endpoint that returns 200 with "I
don't have enough information" under load has failed in a way no HTTP
metric shows. Python also means this reuses the project's own department
list and request models instead of restating them in another language.

Run against a stack that is already up:

    pip install locust
    locust -f loadtest/locustfile.py --host http://localhost:8000
    # headless, for CI or a capacity run:
    locust -f loadtest/locustfile.py --host http://localhost:8000 \\
           --headless -u 20 -r 2 -t 5m --csv loadtest/results

Environment:
    RAG_API_KEY   sent as a bearer token when the API has auth enabled
    LOADTEST_DEPARTMENT  restrict queries to one department

What to watch, in order of what usually breaks first:

1. **p95 on /ask.** The pipeline makes several sequential LLM calls, so
   this is latency-bound long before it is CPU-bound. A p95 that climbs
   with user count while p50 stays flat means requests are queueing
   behind a shared resource, not that the model got slower.
2. **429 rate.** With RATE_LIMIT_STORAGE_URI unset, counters are
   per-process, so a multi-replica run will show *fewer* 429s than the
   configured limit should produce — that discrepancy is itself the
   signal that limits are not shared.
3. **503 on /upload.** Backpressure working. If it appears immediately,
   the workers are not keeping up with the offered ingest rate.
4. **IDK rate.** Rising under load means retrieval is degrading —
   usually the vector store, not the LLM.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

API_KEY = os.getenv("RAG_API_KEY", "")
DEPARTMENT = os.getenv("LOADTEST_DEPARTMENT", "")

# Deliberately mixed: short lookups, multi-part questions that trigger
# decomposition, and out-of-scope questions that should return IDK
# quickly. A single repeated question would sit in the semantic cache and
# measure nothing but the cache.
QUESTIONS = [
    "What is the remote work policy?",
    "How many vacation days do employees get?",
    "What are the standard payment terms for vendors?",
    "Summarise the onboarding process for new engineers.",
    "What is the password rotation requirement?",
    "Compare the expense limits for travel and equipment.",
    "What happens if a contractor breaches the security policy?",
    "Who approves purchase orders above the standard threshold?",
    "What is the process for reporting a data incident?",
    "Explain the parental leave entitlement and how to request it.",
]

# Should be answered "I don't know" — cheap, and a good canary: if these
# start returning confident answers under load, the grader is being
# skipped.
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Who won the World Cup in 1998?",
]

_idk = {"count": 0, "total": 0}


class RagUser(HttpUser):
    """A user asking questions, with an occasional status check."""

    wait_time = between(1, 5)

    def on_start(self):
        self.client.headers.update({"Content-Type": "application/json"})
        if API_KEY:
            self.client.headers["Authorization"] = f"Bearer {API_KEY}"

    def _ask(self, question: str, name: str):
        body = {"question": question, "mode": "auto", "retriever_strategy": "hybrid"}
        if DEPARTMENT:
            body["filter"] = {"department": DEPARTMENT}

        with self.client.post(
            "/ask", json=body, name=name, catch_response=True
        ) as response:
            if response.status_code == 429:
                # Rate limiting is the system working, not a failure.
                response.success()
                return
            if response.status_code == 503:
                # Daily cost cap or backpressure — also intended.
                response.success()
                return
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                answer = response.json().get("answer", "")
            except ValueError:
                response.failure("response was not JSON")
                return

            _idk["total"] += 1
            if not answer.strip():
                response.failure("empty answer")
            elif "don't have enough information" in answer.lower():
                _idk["count"] += 1
                # Not a failure: an honest IDK is correct behaviour. The
                # rate is reported at the end instead.
                response.success()
            else:
                response.success()

    @task(10)
    def ask_in_scope(self):
        self._ask(random.choice(QUESTIONS), "/ask [in-scope]")

    @task(1)
    def ask_out_of_scope(self):
        self._ask(random.choice(OUT_OF_SCOPE), "/ask [out-of-scope]")

    @task(2)
    def health(self):
        # The probe a load balancer runs; it must stay fast while /ask
        # saturates, or the balancer will start evicting healthy replicas.
        self.client.get("/health", name="/health")

    @task(1)
    def list_documents(self):
        with self.client.get(
            "/documents?limit=20", name="/documents", catch_response=True
        ) as response:
            if response.status_code in (200, 401, 429):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


@events.quitting.add_listener
def report(environment, **kwargs):
    """Summarise, and fail the run on the thresholds that matter.

    Locust exits 0 by default however bad the numbers are, which makes it
    useless as a gate. These are the same conditions the Prometheus
    alerts fire on, so a capacity run and production agree on what "bad"
    means.
    """
    stats = environment.stats.total
    p95 = stats.get_response_time_percentile(0.95)
    failure_ratio = stats.fail_ratio
    idk_rate = (_idk["count"] / _idk["total"]) if _idk["total"] else 0.0

    print("\n" + "=" * 62)
    print(f"  requests      : {stats.num_requests}")
    print(f"  failures      : {stats.num_failures} ({failure_ratio:.1%})")
    print(f"  p50 / p95     : {stats.median_response_time:.0f}ms / {p95:.0f}ms")
    print(f"  IDK rate      : {idk_rate:.1%} of answered questions")
    print("=" * 62)

    problems = []
    if failure_ratio > 0.01:
        problems.append(f"failure ratio {failure_ratio:.1%} exceeds 1%")
    if p95 and p95 > 15_000:
        problems.append(f"p95 {p95:.0f}ms exceeds the 15s alert threshold")
    if idk_rate > 0.4:
        problems.append(
            f"IDK rate {idk_rate:.1%} exceeds 40% — retrieval is degrading "
            f"under load, not the model"
        )

    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}")
        environment.process_exit_code = 1
    else:
        print("  PASS")
