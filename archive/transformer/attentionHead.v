module attentionHead #(
    parameter N = 128,
    parameter d = 128,
// scoreMatrix, weightedSum modules assume N and d are multiples of 4
    parameter DATA_WIDTH = 8
// DSP blocks (for multiple/MAC are instantiated as hard-macros of size 8 bit
// if datawidth changes, DSP instantiations need to be updated 
) (
    
    // Input batch size - Nxd, we stream in 1 element(8 bit) at a time
    input [DATA_WIDTH-1:0] data_in,
    input valid_in,
    output ready_in,
    
    input start,  // indicated start of computation for a (Nxd) batch
    input in_last, // indicated last element of input batch of size (Nxd)

    // Output 
    output [DATA_WIDTH-1:0] data_out,
    output valid_out,
    input ready_out
);

// Global controller FSM to manage the flow of data between modules
// 1. Manage input projections, 
    // store all Q, K, V in SRAMs
// 2. Manage dataflow to scoreMatrix
    // - after storing all Q, K, V in SRAMs - start scoreMatrix computations
    // Stream Q in batches of 4 elements (32 bits) over d/4 iterations for 1xd row
    // stream K in batches of 4 rows (4*N*8 bits) over N/4 iterations for Nxd
        // last batch of Q, K - in_last
    // when it is ready for next q input - repeat streaming in Q,K batches over d/4 iterations
// 3. Manage dataflow to softmax
        // valid, ready handshake
        // indicate start of softmax only when weightedSum is not busy - weightedSum is dependent on softmax output
// 4. Manage dataflow to weightedSum
    //  - Once softmax output is valid - start weightedSum computation
        //  softmax output to weightedSum - valid, ready handshake
        // stream in batches of 4 rows of V (4*d*8 bits) er iteration, over N/4 iterations for Nxd
        // last batch of V - in_last
// 5. Output weighted sum when valid
    // - Store in an SRAM or FIFO for top-level output



// Input projections
// 1 DPE each for Q, K, and V

// Q projection - store in regs
    // Have an SRAM of depth=N*(d/4), dataw=32 size, so that we can access all 4 elements of Q for ScoreMatrix in same cycle 
    // have a temp buffer(reg) to hold 4 elements from DPE output and write to SRAM every 4 iterations

// K projection - store in SRAMs
    // Have ‘N’ individual SRAMs of depth=d/4, dataw=32 size. to get 4 columns from 'K' matrx of size (4xN)*8bit , we will be accessing the same row(32 bits) from each of ‘N’ SRAMs.
    // have a temp buffer(reg) to hold 4 elements from DPE output
        // first 4 elements - first SRAM first row
        // second 4 elements - first SRAM second row
        // third 4 elements - first SRAM third row
        //  ......
        // d/4th 4 elements - first SRAM (d/4)th row

        // (d/4 +1)st 4 elements - second SRAM first row
        // (d/4 +2)nd 4 elements - second SRAM second row
        // ......


// V projection - store in SRAMs
    // have 'd' SRAMs of depth=N, dataw=8 size - no buffer needed, can directly write to SRAMs
          // for each iteration we read 4 rows from each of 'd' SRAMs (takes 4 cycles)
          // we can then set the  v_valid_in for weightedSum
          // this matches the 4 rows of V needed for weightedSum in same cycle



// Score matrix module instantiation
// Expects Q: 4 elements per cycle (32 bits), cover a 1xd row in d/4 cycles
// Expects K: 4 full rows per cycle (4*N*8 bits), cover entire Nxd matrix in N/4 cycles
// start signal at start of a new Q token(1xd)
//  in_last for last batch - last 1/4 elements of Q/ last 4 rows of K
// Outputs: N*8 exp scores and a  8-bit 1/(sum of scores)
    scoreMatrix #(
        .N(N),
        .d(d),
        .DATA_WIDTH(DATA_WIDTH)
    ) scoreMatrix_inst (
        // Q input of size 4x1 per batch (4 dims)
        .q_data_in(), 
        .q_valid_in(),
        .q_ready_in(),

        // K input of size 4xN per batch (4 full rows)
        .k_data_in(), 
        .k_valid_in(),
        .k_ready_in(),

        .start(start),  // indicated start of computation for a (Nxd) batch
        .in_last(in_last), // indicate last batch of K rows

        // Output score matrix of size (1xN)
        .score_data_out(), // size - N*DATA_WIDTH
        .score_sum_out(), // output 1/sum of exp scores for softmax normalization
        .score_valid_out(),
        .score_ready_out()
    );


// Softmax module instantiation
// Expects: N*8 exp scores and  8-bit 1/sum
// start pulse after scoreMatrix (we do start this when weightedSum is processing as it depends on softmax output )
// Outputs: N*8 normalized probs

    softmax #(
        .N(N),
        .d(d),
        .DATA_WIDTH(DATA_WIDTH)
    ) softmax_inst (
        .score_data_in(), // size - N*DATA_WIDTH
        .score_sum_in(), // 1/sum of all scores for normalization
        .score_valid_in(),
        .score_ready_in(),

        .start(),  // indicates start of computation

        // Softmax output of size (1xN)
        .softmax_data_out(), // size - N*DATA_WIDTH
        .softmax_valid_out(),
        .softmax_ready_out()
    );


// Weighted Sum module instantiation
// Expects: N*8 softmax results
// expects V: 4 rows per iteration (4*d)*8bit, over N/4 iterations
// start: After softmax is done
//  in_last for Last batch of 'V'
// Outputs: d*8 attention result
    weightedSum #(
        .N(N),
        .d(d),
        .DATA_WIDTH(DATA_WIDTH)
    ) weightedSum_inst (
        .softmax_data_in(), // size - N*DATA_WIDTH for (1xN)
        .softmax_valid_in(),
        .softmax_ready_in(),

        // V inputs, 4 rows each of size (1xd)
        .v0_data_in(), // d*DATA_WIDTH
        .v1_data_in(), // d*DATA_WIDTH
        .v2_data_in(), // d*DATA_WIDTH
        .v3_data_in(), // d*DATA_WIDTH
        .v_valid_in(),
        .v_ready_in(),

        .start(start),  // indicated start of computation for a (Nxd) batch
        .in_last(in_last), // indicate last batch of V of size (Nxd)

        // Output weighted sum of size (1xd)
        .weighted_sum_out(), // size - d*DATA_WIDTH
        .valid_out(),
        .ready_out()
    );


endmodule