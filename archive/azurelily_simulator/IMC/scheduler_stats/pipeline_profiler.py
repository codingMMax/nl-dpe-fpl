from __future__ import annotations


class PipelineProfiler:
    """
    Cross-layer critical-path profiler.

    This profiler receives per-layer timing tuples:
      - first_output_ns
      - steady_ns
      - events
      - required_upstream_outputs

    and accumulates critical-path latency contributions.
    """

    def __init__(self):
        self._pending_parallel = {"attn_qkv": {}}
        self._prev_output_ready = []
        self._prev_events = 0
        self._prev_finish = 0.0
        self._critical_end = 0.0
        self._trace = []

    def _required_upstream_index(self, layer, token_idx: int, required_upstream_outputs: int) -> int:
        prev_events = len(self._prev_output_ready)
        if prev_events <= 0:
            return 1

        if layer is not None and layer.type in ("conv2d", "maxpool"):
            positions_per_input = max(1, layer.output_height * layer.output_width)
            batch_idx = (token_idx - 1) // positions_per_input
            token_in_batch = (token_idx - 1) % positions_per_input
            out_h = token_in_batch // layer.output_width
            out_w = token_in_batch % layer.output_width

            input_start_h = out_h * layer.stride - layer.padding
            input_start_w = out_w * layer.stride - layer.padding
            max_h = min(layer.input_height - 1, input_start_h + layer.kernel_size - 1)
            max_w = min(layer.input_width - 1, input_start_w + layer.kernel_size - 1)
            max_h = max(0, max_h)
            max_w = max(0, max_w)

            required_in_batch = max_h * layer.input_width + max_w + 1
            prev_positions_per_input = max(1, layer.input_height * layer.input_width)
            required = batch_idx * prev_positions_per_input + required_in_batch
            return min(max(1, required), prev_events)

        fallback = required_upstream_outputs + token_idx - 1
        return min(max(1, fallback), prev_events)

    def _build_output_ready_times(
        self,
        layer,
        value: float,
        first_output_ns: float,
        steady_ns: float,
        events: int,
        required_upstream_outputs: int,
    ):
        if events <= 1:
            if self._prev_events > 1 and self._prev_output_ready:
                req_idx = self._required_upstream_index(layer, 1, required_upstream_outputs)
                launch = self._prev_output_ready[req_idx - 1]
            else:
                launch = self._prev_finish
            finish = launch + value
            return [finish], launch, finish, finish, 0.0

        if self._prev_events > 1 and self._prev_output_ready:
            req_idx = self._required_upstream_index(layer, 1, required_upstream_outputs)
            launch = self._prev_output_ready[req_idx - 1]
        else:
            launch = self._prev_finish

        first_ready = launch + first_output_ns
        out_ready = [first_ready]
        backpressure_add = 0.0
        for token_idx in range(2, events + 1):
            if self._prev_events > 1 and self._prev_output_ready:
                req_idx = self._required_upstream_index(layer, token_idx, required_upstream_outputs)
                upstream_ready = self._prev_output_ready[req_idx - 1]
            else:
                upstream_ready = self._prev_finish

            local_ready = out_ready[-1] + steady_ns
            dependency_ready = upstream_ready + first_output_ns
            out_t = max(local_ready, dependency_ready)
            if dependency_ready > local_ready:
                backpressure_add += dependency_ready - local_ready
            out_ready.append(out_t)

        finish = max(out_ready[-1], launch + value)
        out_ready[-1] = finish
        return out_ready, launch, first_ready, finish, backpressure_add

    def _record_single(
        self,
        key: str,
        value: float,
        first_output_ns: float | None = None,
        steady_ns: float | None = None,
        events: int = 1,
        required_upstream_outputs: int = 1,
        layer=None,
    ) -> dict[str, float]:
        if first_output_ns is None:
            first_output_ns = value
        first_output_ns = max(0.0, min(first_output_ns, value))

        if steady_ns is None:
            steady_ns = value
        steady_ns = max(0.0, steady_ns)

        events = max(1, int(events))
        required_upstream_outputs = max(1, int(required_upstream_outputs))
        required_eff = required_upstream_outputs if self._prev_events > 1 else 1
        out_ready, start, first_ready, finish, backpressure_add = self._build_output_ready_times(
            layer,
            value,
            first_output_ns,
            steady_ns,
            events,
            required_eff,
        )
        contribution = max(0.0, finish - self._critical_end)
        self._critical_end = max(self._critical_end, finish)
        self._prev_output_ready = out_ready
        self._prev_events = events
        self._prev_finish = finish
        self._trace.append(
            {
                "layer": key,
                "layer_type": None if layer is None else layer.type,
                "events": events,
                "required_upstream_outputs": required_eff,
                "start_ns": start,
                "first_output_ns": first_output_ns,
                "first_output_t_ns": first_ready,
                "steady_ns": steady_ns,
                "base_total_ns": value,
                "backpressure_add_ns": backpressure_add,
                "effective_total_ns": finish - start,
                "finish_ns": finish,
                "critical_contribution_ns": contribution,
                "critical_end_ns": self._critical_end,
            }
        )
        return {key: contribution}

    def _flush_pending_parallel(self) -> dict[str, float]:
        out = {}
        pending = self._pending_parallel.get("attn_qkv", {})
        if pending:
            parallel_lat = max(pending.values())
            out.update(
                self._record_single(
                    "attn_qkv_parallel",
                    parallel_lat,
                    first_output_ns=parallel_lat,
                    steady_ns=parallel_lat,
                    events=1,
                    required_upstream_outputs=1,
                    layer=None,
                )
            )
            pending.clear()
        return out

    def record(self, layer, value: float, timing: dict | None = None) -> dict[str, float]:
        out = {}
        if layer.name in ("linear_Q", "linear_K", "linear_V"):
            pending = self._pending_parallel["attn_qkv"]
            pending[layer.name] = pending.get(layer.name, 0.0) + value
            if len(pending) == 3:
                out.update(self._flush_pending_parallel())
            return out

        out.update(self._flush_pending_parallel())
        first_output_ns = None
        steady_ns = None
        events = 1
        required_upstream_outputs = 1
        if timing is not None:
            first_output_ns = timing.get("first_output_ns", timing.get("fill"))
            steady_ns = timing.get("steady")
            events = timing.get("events", 1)
            required_upstream_outputs = timing.get("required_upstream_outputs", 1)
        out.update(
            self._record_single(
                layer.name,
                value,
                first_output_ns=first_output_ns,
                steady_ns=steady_ns,
                events=events,
                required_upstream_outputs=required_upstream_outputs,
                layer=layer,
            )
        )
        return out

    def finalize(self) -> dict[str, float]:
        return self._flush_pending_parallel()

    def trace(self):
        return list(self._trace)
