`define DWIDTH    40
`define LOGDWIDTH 6
`define AWIDTH    9
`define MEM_SIZE  512

`define CMD_ADDR   9'b111111111
`define ALU_CMD    4'b0000
`define COPY_CMD   4'b0001
`define LSHIFT_CMD 4'b0010
`define RSHIFT_CMD 4'b0011
`define NOT_CMD 4'b0100
`define AND_CMD 4'b0101
`define XOR_CMD 4'b0110
`define OR_CMD 4'b0111

module compute_ram_wrapper (
        //write port
        addr1, 
        d1, 
        we1, 
        //read port
        addr2, 
        q2,  
        //direct interconnect
        pe_in,
        pe_out,
        clk);

input [`AWIDTH-1:0] addr1;
input [`DWIDTH-1:0] d1;
input we1;

input [`AWIDTH-1:0] addr2;
output [`DWIDTH-1:0] q2;

input pe_in;
output reg pe_out;
input clk;

`ifdef VCS
wire [`DWIDTH-1:0] d;
assign d = d1;
wire we;
assign we = we1;
reg [`DWIDTH-1:0] q;
assign q2 = q;
wire [`AWIDTH-1:0] addr;
assign addr = we ? addr1 : addr2;


//ram that matches external interface
reg [`DWIDTH-1:0] ram[((1<<`AWIDTH)-1):0];

//ram that is based on the internal configuration
//a 160x128 ram
reg [159:0] ram_internal[127:0];

wire compute_mode;
assign compute_mode = (addr == `CMD_ADDR);

/////////////////////////////////////////////////////
// Compute RAM behavioral model
/////////////////////////////////////////////////////
//If Address is `CMD_ADDR, then the data contains the command.

//Let's say the structure of the commnd is:
//<PREDICATE> <WRITE_SEL> <PORT> <C_EN> <T_EN> <ALU_TRUTH_TABLE> <dst_row>, <src2_row>, <src1_row>
//   2 bits     2 bits     1 bit  1 bit  1 bit     4 bits          7 bits     7 bits      7 bits

//Row addresses are 7 bits because the organization is 128x128
//In some cases like COPY or SHIFT, one of the
//src rows will be blank.

//Example:
//TAG <> <> <> <> <> ALU_CMD_FOR_ADD 125, 113, 191 
//This command will run the operation stored in the ALU
//if TAG is 1 
//If ALU_CMD_FOR_ADD corresponds to the specific truth table values
//to perform SUM, this command will add contents of row 191 and row 113
//and write the result to row 125, if TAG latch was 1.

wire [1:0] predicate;
wire [1:0] write_sel;
wire port;
wire c_en;
wire t_en;
wire [3:0] truth_table;
wire [6:0] dst;
wire [6:0] src2;
wire [6:0] src1;

assign predicate = d[39:38];
assign write_sel = d[29:28];
assign port = d[27];
assign c_en      = d[26];
assign t_en      = d[25];
assign truth_table    = d[24:21];
assign dst       = d[20:14];
assign src2      = d[13:7];
assign src1      = d[6:0];

//there is one carry latch/ff in each physical ram column
reg [160:0] carry;

//temporary
reg [159:0] temp[127:0];
integer i;
//behavioral reset
initial begin
  carry = 0;
  for (i=0; i<160; i = i +1) begin
    temp[i] = 0;
  end
end

wire [5:0] command;
assign command = {write_sel, truth_table};

always @(posedge clk) begin 
  //compute mode
  if (compute_mode) begin

    //Look at what the truth_table is and operate accordingly
    case (truth_table) 
      //Just modelling this as if ALU_CMD configured into
      //the compute_ram is actually ADD
      `ALU_CMD : begin          
        //if predicate condition is true, then...
        temp[dst] = ram_internal[src2]^ram_internal[src1]^carry;
        if(c_en) begin
          carry <= (ram_internal[src2]&ram_internal[src1])|(ram_internal[src2]&carry)|(ram_internal[src1]&carry);
        end
        else begin
          carry <= 0; 
        end
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
      end
      `COPY_CMD : begin          
        //if predicate condition is true, then...
        temp[dst] = ram_internal[src1];
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
      end
      `LSHIFT_CMD: begin
        //if predicate condition is true, then...
        temp[dst] = {ram_internal[src1][158:0], pe_in};
        pe_out <= ram_internal[src1][127];
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
      end
      `RSHIFT_CMD: begin
        //if predicate condition is true, then...
        temp[dst] = {pe_in,ram_internal[src1][159:1]};
        pe_out <= ram_internal[src1][0];
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
      end
    `NOT_CMD: begin
      temp[dst] = ~ram_internal[src1];
       
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
    end
    `AND_CMD: begin
      temp[dst] = ram_internal[src1]&ram_internal[src2];
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
    end
    `XOR_CMD: begin
      temp[dst] = ram_internal[src1]^ram_internal[src2];
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
    end
    `OR_CMD: begin
      temp[dst] = ram_internal[src1]|ram_internal[src2];
        //update ram_internal and ram
        ram_internal[dst] <= temp[dst];
        ram[(dst<<2)+0] <= temp[dst][31:0];
        ram[(dst<<2)+1] <= temp[dst][63:32];
        ram[(dst<<2)+2] <= temp[dst][95:64];
        ram[(dst<<2)+3] <= temp[dst][127:96];
        ram[(dst<<2)+4] <= temp[dst][159:128];
    end
      //Not modelling anything else for now
      default : begin
        //$display("%h command is not modelled", command);
      end
    endcase

  end

  //memory mode
  else begin 

    if (we) begin
      ram[addr] <= d;
      //Also update ram_internal
      case (addr[1:0]) 
      2'b00 : ram_internal[addr[`AWIDTH-1:2]][1*`DWIDTH-1:0*`DWIDTH] <= d;
      2'b01 : ram_internal[addr[`AWIDTH-1:2]][2*`DWIDTH-1:1*`DWIDTH] <= d;
      2'b10 : ram_internal[addr[`AWIDTH-1:2]][3*`DWIDTH-1:2*`DWIDTH] <= d;
      2'b11 : ram_internal[addr[`AWIDTH-1:2]][4*`DWIDTH-1:3*`DWIDTH] <= d;
      endcase
    end
    else begin
      q <= ram[addr];
    end

  end  

end

`else

compute_ram_simple_dp u_compute_ram(
.addr1(addr1),
.we1(we1),
.data1(d1),
.addr2(addr2),
.out2(q2),
.pe_in(pe_in),
.pe_out(pe_out),
.clk(clk)
);

`endif

endmodule