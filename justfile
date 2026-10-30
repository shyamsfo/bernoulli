# justfile — Bernoulli OSS core
#
# Dev + benchmark recipes for the open-source library. Operational recipes
# (serve-up / serve-start / deploy / website publishing) live in the private
# sister repo at shyamsfo/bernoulli-ops.
#
# See `just --list dev` for code quality + local serving + Docker + model-pull,
# and `just --list bench` for the eval harness + benchmark sweeps + load test.

mod dev
mod bench

# Default: show the task list
default:
    @just --list
