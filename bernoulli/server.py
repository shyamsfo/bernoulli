"""FastAPI server exposing /v1/decide, /healthz, /v1/models.

Lands in M4. Batches `questions x permutations` into a single engine call
against a shared state prefix. Pydantic models for request/response live in
bernoulli.types.
"""
