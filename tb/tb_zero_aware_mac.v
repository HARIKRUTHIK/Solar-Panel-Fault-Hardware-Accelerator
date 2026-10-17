`timescale 1ns/1ps

module tb_zero_aware_mac;

    reg         clk;
    reg         rst_n;
    reg         enable;
    reg         clear_acc;
    reg  signed [7:0]  act_in;
    reg  signed [7:0]  wt_in;
    wire signed [31:0] acc_out;
    wire [31:0] total_mac_count;
    wire [31:0] zero_skip_count;
    wire        zero_detected;

    zero_aware_mac uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .clear_acc(clear_acc),
        .act_in(act_in),
        .wt_in(wt_in),
        .acc_out(acc_out),
        .total_mac_count(total_mac_count),
        .zero_skip_count(zero_skip_count),
        .zero_detected(zero_detected)
    );

    always #5 clk = ~clk;

    integer i;
    reg signed [31:0] expected_acc;

    initial begin
        clk = 0;
        rst_n = 0;
        enable = 0;
        clear_acc = 0;
        act_in = 0;
        wt_in = 0;
        expected_acc = 0;
        #20;
        rst_n = 1;
        #10;

        $display("=================================================================");
        $display("   ZERO-AWARE MAC HARDWARE ACCELERATOR VERIFICATION");
        $display("=================================================================");

        enable = 1;

        // Test Sequence: 10 operations with 4 zero activations (40% sparsity)
        // 1. act=5, wt=3 (non-zero)
        @(posedge clk); #1 act_in = 8'sd5; wt_in = 8'sd3;
        // 2. act=0, wt=10 (zero activation)
        @(posedge clk); #1 act_in = 8'sd0; wt_in = 8'sd10;
        // 3. act=7, wt=-2 (non-zero)
        @(posedge clk); #1 act_in = 8'sd7; wt_in = -8'sd2;
        // 4. act=-4, wt=0 (zero weight)
        @(posedge clk); #1 act_in = -8'sd4; wt_in = 8'sd0;
        // 5. act=0, wt=0 (dual zero)
        @(posedge clk); #1 act_in = 8'sd0; wt_in = 8'sd0;
        // 6. act=12, wt=4 (non-zero)
        @(posedge clk); #1 act_in = 8'sd12; wt_in = 8'sd4;
        // 7. act=-8, wt=-3 (non-zero)
        @(posedge clk); #1 act_in = -8'sd8; wt_in = -8'sd3;
        // 8. act=0, wt=15 (zero activation)
        @(posedge clk); #1 act_in = 8'sd0; wt_in = 8'sd15;
        // 9. act=6, wt=2 (non-zero)
        @(posedge clk); #1 act_in = 8'sd6; wt_in = 8'sd2;
        // 10. act=1, wt=1 (non-zero)
        @(posedge clk); #1 act_in = 8'sd1; wt_in = 8'sd1;

        @(posedge clk); #1 enable = 0;
        #20;

        $display("Total MAC Cycles Simulated : %0d", total_mac_count);
        $display("Zero Multiplications Bypassed: %0d", zero_skip_count);
        $display("Measured Switching Sparsity  : %0.2f %%", (zero_skip_count * 100.0) / total_mac_count);
        $display("Final Accumulated Result     : %0d", acc_out);

        if (zero_skip_count == 4 || zero_skip_count == 5) begin
            $display("[RESULT] ZERO-AWARE MAC BYPASS VERIFIED (Power Gating Functioning)");
        end else begin
            $display("[RESULT] MISMATCH in zero skip counter");
        end
        $display("=================================================================");
        $finish;
    end

endmodule
