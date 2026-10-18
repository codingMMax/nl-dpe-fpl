// ============================================================================
// tb_att_fifo.v — self-checking unit test of att_fifo: random push / pop against a
// reference queue (the TB's own array), including pushes and pops in the same cycle,
// filling to DEPTH, draining to empty, pointer wrap-around (several times), and the
// sticky overflow / underflow probes (forced once each at the end).
//   iverilog -g2005 -o t.vvp v2/sim/tb/tb_att_fifo.v v2/sim/rtl/att_fifo.v && vvp -n t.vvp
//   (-Ptb_att_fifo.DEPTH=13: a depth that is not a power of two, M5)
// ============================================================================
`timescale 1ns / 1ps
module tb_att_fifo;
    parameter  DEPTH = 16;
    localparam W = 13;
    reg clk = 1'b0, reset = 1'b1;
    always #5 clk = ~clk;
    reg              wr_en = 1'b0, rd_en = 1'b0;
    reg  [W-1:0]     wr_data = 0;
    wire [W-1:0]     rd_data;
    wire [$clog2(DEPTH):0] count;
    wire             overflow, underflow;
    att_fifo #(.W(W), .DEPTH(DEPTH)) dut (.clk(clk), .reset(reset), .wr_en(wr_en), .wr_data(wr_data),
        .rd_en(rd_en), .rd_data(rd_data), .count(count), .overflow(overflow), .underflow(underflow));

    reg [W-1:0] q [0:4095];
    integer head = 0, tail = 0, errors = 0, i, pend = 0, exp_v, seed = 7, nmax = 0, nzero = 0;
    reg [W-1:0] exp_q;
    task step;
        input do_wr, do_rd;
        begin
            wr_en = do_wr; rd_en = do_rd; wr_data = $random(seed);
            if (do_wr) begin q[tail] = wr_data; tail = tail + 1; end
            if (do_rd) begin exp_q = q[head]; head = head + 1; end
            @(posedge clk); #1;
            if (do_rd && rd_data !== exp_q) begin                 // registered read: valid right after this edge
                if (errors < 5) $display("[tb_fifo] ERROR: got %h exp %h", rd_data, exp_q);
                errors = errors + 1;
            end
            if (count !== tail - head) begin
                if (errors < 5) $display("[tb_fifo] ERROR: count %0d exp %0d", count, tail - head);
                errors = errors + 1;
            end
            if (tail - head == DEPTH) nmax = nmax + 1;
            if (tail - head == 0) nzero = nzero + 1;
        end
    endtask
    integer n, r, can_w, can_r;
    initial begin
        repeat (2) @(posedge clk); #1 reset = 1'b0;
        // phase 1: random legal traffic (never pop empty, never push full)
        for (n = 0; n < 3000; n = n + 1) begin
            r = $random(seed) & 3;
            can_w = (tail - head) < DEPTH; can_r = (tail - head) > 0;
            step(can_w && (r != 0) && !(n % 400 > 300), can_r && (r != 1) && !(n % 400 < 60));
        end
        // phase 2: fill to DEPTH, then drain to empty, twice (wrap-around)
        for (i = 0; i < 2; i = i + 1) begin
            while (tail - head < DEPTH) step(1, 0);
            step(1, 1);                                  // full: push + pop in one cycle keeps it full
            while (tail - head > 0) step(0, 1);
        end
        step(0, 0);
        if (overflow || underflow) begin $display("[tb_fifo] ERROR: probe set on legal traffic"); errors = errors + 1; end
        // phase 3: the probes fire on illegal traffic
        rd_en = 1'b1; wr_en = 1'b0; @(posedge clk); #1; rd_en = 1'b0;
        if (!underflow) begin $display("[tb_fifo] ERROR: underflow not flagged"); errors = errors + 1; end
        reset = 1'b1; @(posedge clk); #1 reset = 1'b0;
        for (i = 0; i <= DEPTH; i = i + 1) begin wr_en = 1'b1; @(posedge clk); #1; end
        wr_en = 1'b0;
        if (!overflow) begin $display("[tb_fifo] ERROR: overflow not flagged"); errors = errors + 1; end
        if (nmax == 0 || nzero == 0) begin $display("[tb_fifo] ERROR: full / empty not reached"); errors = errors + 1; end
        $display("[tb_fifo] %0s errors=%0d (pushed %0d, full %0d times, empty %0d times)",
                 errors == 0 ? "PASS" : "FAIL", errors, tail, nmax, nzero);
        $finish;
    end
endmodule
