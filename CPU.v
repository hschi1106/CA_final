// Your code
module CPU(clk,
            rst_n,
            // For mem_D (data memory)
            wen_D,
            addr_D,
            wdata_D,
            rdata_D,
            // For mem_I (instruction memory (text))
            addr_I,
            rdata_I);

    input         clk, rst_n ;
    // For mem_D
    output        wen_D  ;
    output [31:0] addr_D ;
    output [31:0] wdata_D;
    input  [31:0] rdata_D;
    // For mem_I
    output [31:0] addr_I ;
    input  [31:0] rdata_I;
    
    //---------------------------------------//
    // Do not modify this part!!!            //
    // Exception: You may change wire to reg //
    reg    [31:0] PC          ;              //
    wire   [31:0] PC_nxt      ;              //
    wire          regWrite    ;              //
    wire   [ 4:0] rs1, rs2, rd;              //
    wire   [31:0] rs1_data    ;              //
    wire   [31:0] rs2_data    ;              //
    wire   [31:0] rd_data     ;              //
    //---------------------------------------//

    // Instruction fields
    wire [31:0] instr = rdata_I;
    wire [6:0]  opcode = instr[6:0];
    wire [4:0]  reg_rd = instr[11:7];
    wire [2:0]  funct3 = instr[14:12];
    wire [4:0]  reg_rs1 = instr[19:15];
    wire [4:0]  reg_rs2 = instr[24:20];
    wire [6:0]  funct7 = instr[31:25];

    // Control signals
    wire        Branch, MemRead, MemtoReg, MemWrite, ALUSrc, ALUregWrite, Jump, Jalr;
    wire [3:0]  ALU_control;
    wire        mulDiv_valid, mulDiv_ready;
    wire [1:0]  mulDiv_control;
    assign regWrite = ALUregWrite || mulDiv_ready;

    // Immediate generation
    wire [31:0] imm;

    // ALU
    wire [31:0] ALU_in1;
    wire [31:0] ALU_in2;
    wire [31:0] ALU_result;
    wire        ALU_zero;

    // MulDiv
    wire [31:0] MulDiv_result;

    // PC related
    wire [31:0] PC_plus4 = PC + 32'd4;
    wire [31:0] branch_jal_target;
    wire [31:0] jalr_target;

    // Write back data
    wire [31:0] mem_data_out = rdata_D;
    wire [31:0] wb_data = MemtoReg     ? mem_data_out : 
                          ALUregWrite  ? ALU_result :
                          mulDiv_ready ? MulDiv_result :
                          32'b0;
    assign rd_data = (Jump || Jalr) ? PC_plus4 : wb_data;

    
    //==========================
    // Register file connection
    //==========================
    assign rs1 = reg_rs1;
    assign rs2 = reg_rs2;
    assign rd  = reg_rd;

    //---------------------------------------//
    // Do not modify this part!!!            //
    reg_file reg0(                           //
        .clk(clk),                           //
        .rst_n(rst_n),                       //
        .wen(regWrite),                      //
        .a1(rs1),                            //
        .a2(rs2),                            //
        .aw(rd),                             //
        .d(rd_data),                         //
        .q1(rs1_data),                       //
        .q2(rs2_data));                      //
    //---------------------------------------//

    //==========================
    // Immediate Generation
    //==========================
    ImmediateGenerator immGen(
        .instr(instr),
        .imm(imm)
    );

    //==========================
    // ALU
    //==========================
    assign ALU_in1 = ALU_control == 4'b0111 ? PC : rs1_data;
    assign ALU_in2 = ALUSrc ? imm : rs2_data;
    ALU alu0(
        .in1(ALU_in1),
        .in2(ALU_in2),
        .control(ALU_control),
        .result(ALU_result),
        .zero(ALU_zero)
    );

    //==========================
    // Branch and Jump targets
    //==========================
    assign branch_jal_target = PC + imm;            // branch and jal target
    assign jalr_target = (rs1_data + imm) & ~32'h1; // jalr target

    //==========================
    // Next PC Logic
    //==========================
    wire stall = mulDiv_valid;
    wire signed_lt = ($signed(ALU_result) < 0); 
    wire branch_taken = Branch & (
       (funct3 == 3'b000 & ALU_zero)  | // beq
       (funct3 == 3'b001 & ~ALU_zero) | // bne
       (funct3 == 3'b100 & signed_lt) | // blt
       (funct3 == 3'b101 & ~signed_lt)  // bge
    );

    assign PC_nxt = (stall)        ? PC : 
                    (Jump)         ? branch_jal_target :
                    (Jalr)         ? jalr_target :
                    (branch_taken) ? branch_jal_target :
                    PC_plus4;

    //==========================
    // Data Memory Interface
    //==========================
    assign wen_D = MemWrite;
    assign addr_D  = ALU_result;
    assign wdata_D = rs2_data;

    //==========================
    // Instruction Memory Interface
    //==========================
    assign addr_I = PC;

    //==========================
    // Control Unit
    //==========================
    ControlUnit ctrl(
        .opcode(opcode),
        .funct3(funct3),
        .funct7(funct7),
        .mulDiv_ready(mulDiv_ready),
        .ALUregWrite(ALUregWrite),
        .ALUSrc(ALUSrc),
        .MemRead(MemRead),
        .MemWrite(MemWrite),
        .MemtoReg(MemtoReg),
        .Branch(Branch),
        .Jump(Jump),
        .Jalr(Jalr),
        .ALUOp(ALU_control),
        .mulDiv_valid(mulDiv_valid),
        .mulDivOp(mulDiv_control)
    );

    //==========================
    // mulDiv Unit
    //==========================
    MulDiv mulDiv0(
        .clk(clk),
        .rst_n(rst_n),
        .valid(mulDiv_valid),
        .ready(mulDiv_ready),
        .control(mulDiv_control),
        .in_A(rs1_data),
        .in_B(rs2_data),
        .out(MulDiv_result)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            PC <= 32'h00010000; // Do not modify this value!!!
            
        end
        else begin
            PC <= PC_nxt;
            
        end
    end
endmodule

// Do not modify the reg_file!!!
module reg_file(clk, rst_n, wen, a1, a2, aw, d, q1, q2);

    parameter BITS = 32;
    parameter word_depth = 32;
    parameter addr_width = 5; // 2^addr_width >= word_depth

    input clk, rst_n, wen; // wen: 0:read | 1:write
    input [BITS-1:0] d;
    input [addr_width-1:0] a1, a2, aw;

    output [BITS-1:0] q1, q2;

    reg [BITS-1:0] mem [0:word_depth-1];
    reg [BITS-1:0] mem_nxt [0:word_depth-1];

    integer i;

    assign q1 = mem[a1];
    assign q2 = mem[a2];

    always @(*) begin
        for (i=0; i<word_depth; i=i+1)
            mem_nxt[i] = (wen && (aw == i)) ? d : mem[i];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1) begin
                case(i)
                    32'd2: mem[i] <= 32'hbffffff0;
                    32'd3: mem[i] <= 32'h10008000;
                    default: mem[i] <= 32'h0;
                endcase
            end
        end
        else begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1)
                mem[i] <= mem_nxt[i];
        end
    end
endmodule

module ControlUnit(
    input [6:0] opcode,
    input [2:0] funct3,
    input [6:0] funct7,
    input mulDiv_ready,
    output reg ALUregWrite,
    output reg ALUSrc,
    output reg MemRead,
    output reg MemWrite,
    output reg MemtoReg,
    output reg Branch,
    output reg Jump,
    output reg Jalr,
    output reg [3:0] ALUOp,
    output reg mulDiv_valid,
    output reg [1:0] mulDivOp);
    // opcode:
    // Load:    0000011 (lw)
    // Store:   0100011 (sw)
    // Imm ALU: 0010011 (addi, slti, sltiu, xori, ori, andi, slli, srli, srai)
    // Reg ALU: 0110011 (add, sub, sll, srl, sra, xor, or, and, slt, sltu, mul, divu, remu)
    // Branch:  1100011 (beq, bne, bge, blt)
    // JAL:     1101111 (jal)
    // JALR:    1100111 (jalr)
    // LUI:     0110111 (lui)
    // AUIPC:   0010111 (auipc)

    // mul/div/remu:
    wire isMul  = (opcode == 7'b0110011 && funct7 == 7'b0000001 && funct3 == 3'b000);
    wire isDivu = (opcode == 7'b0110011 && funct7 == 7'b0000001 && funct3 == 3'b101);
    wire isRemu = (opcode == 7'b0110011 && funct7 == 7'b0000001 && funct3 == 3'b111);

    always @(*) begin
        ALUregWrite     = 0;
        ALUSrc          = 0;
        MemRead         = 0;
        MemWrite        = 0;
        MemtoReg        = 0;
        Branch          = 0;
        Jump            = 0;
        Jalr            = 0;
        ALUOp           = 4'b0000;
        mulDiv_valid    = 0;
        mulDivOp     = 2'b00;

        case(opcode)
            7'b0110011: begin // R-type arithmetic (add, sub, mul, divu, remu)
                if (funct7 == 7'b0000000) begin
                    // add
                    ALUregWrite = 1;
                    ALUSrc   = 0;
                    case(funct3)
                        3'b000: ALUOp = 4'b0001; // add
                        default: ALUOp = 4'b0000;
                    endcase
                end else if (funct7 == 7'b0100000) begin
                    // sub
                    ALUregWrite = 1;
                    ALUSrc   = 0;
                    case(funct3)
                        3'b000: ALUOp = 4'b0010; // sub
                        default: ALUOp = 4'b0000;
                    endcase
                end else if (funct7 == 7'b0000001) begin
                    // mul/divu/remu
                    ALUregWrite = 0;
                    mulDiv_valid = ~mulDiv_ready;
                    // ALUOp對muldDiv無作用,保持0000即可
                    if(isMul)  mulDivOp = 2'b00; // mul
                    else if(isDivu) mulDivOp = 2'b01; // divu
                    else if(isRemu) mulDivOp = 2'b10; // remu
                end
            end

            7'b0010011: begin // I-type arithmetic (addi, slli, srli, srai, slti)
                ALUregWrite = 1;
                ALUSrc   = 1;
                case(funct3)
                    3'b000: ALUOp = 4'b0001; // addi
                    3'b001: ALUOp = 4'b0011; // slli
                    3'b101: ALUOp = (funct7[5]) ? 4'b0101 : 4'b0100; // srai/srli
                    3'b010: ALUOp = 4'b0110; // slti
                    default: ALUOp = 4'b0000;
                endcase
            end

            7'b0000011: begin // Load (lw)
                ALUregWrite = 1;
                ALUSrc   = 1;
                MemRead  = 1;
                MemtoReg = 1;
                ALUOp    = 4'b0001; // add for address calc
            end

            7'b0100011: begin // Store (sw)
                ALUSrc   = 1;
                MemWrite = 1;
                ALUOp    = 4'b0001; // add for address calc
            end

            7'b1100011: begin // Branch (beq, bne, bge, blt)
                Branch = 1;
                ALUOp  = 4'b0010; // sub for branch calc
            end

            7'b1101111: begin // jal
                ALUregWrite = 1;
                Jump     = 1;
                ALUOp    = 4'b0001; // add for address calc
            end

            7'b1100111: begin // jalr
                ALUregWrite = 1;
                Jalr     = 1;
                ALUSrc   = 1;
                ALUOp    = 4'b0001; // add for address calc
            end

            7'b0110111: begin // lui
                ALUregWrite = 1;
                ALUSrc   = 1;
                ALUOp    = 4'b1000; 
            end

            7'b0010111: begin // auipc
                ALUregWrite = 1;
                ALUSrc   = 1;
                ALUOp    = 4'b0111;
            end

            default: begin
                // no-OP or unknown opcode
            end
        endcase
    end
endmodule

module ImmediateGenerator(
    input [31:0] instr,
    output reg [31:0] imm);
    wire [6:0] opcode = instr[6:0];
    wire [2:0] funct3 = instr[14:12];

    always @(*) begin
        case(opcode)
            // I-type
            7'b0000011, // lw
            7'b0010011, // addi, slli, srli, srai, slti
            7'b1100111: // jalr
                imm = {{20{instr[31]}}, instr[31:20]};

            // S-type
            7'b0100011: // sw
                imm = {{20{instr[31]}}, instr[31:25], instr[11:7]};

            // B-type
            7'b1100011: // beq, bne, bge, blt
                imm = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};

            // U-type
            7'b0010111, // auipc
            7'b0110111: // lui
                imm = {instr[31:12], 12'b0};

            // J-type
            7'b1101111: // jal
                imm = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

            default: imm = 32'b0;
        endcase
    end
endmodule

module ALU(
    input  [31:0] in1,
    input  [31:0] in2,
    input  [3:0]  control,
    output reg [31:0] result,
    output zero);
    // ALU control:
    // 0000: error or no-OP
    // 0001: add
    // 0010: sub
    // 0011: sll
    // 0100: srl
    // 0101: sra
    // 0110: slti
    // 0111: auipc
    // 1000: lui

    always @(*) begin
        case(control)
            4'b0001: result = in1 + in2;                                    // add
            4'b0010: result = in1 - in2;                                    // sub
            4'b0011: result = in1 << in2[4:0];                              // sll
            4'b0100: result = in1 >> in2[4:0];                              // srl
            4'b0101: result = $signed(in1) >>> in2[4:0];                    // sra
            4'b0110: result = $signed(in1) < $signed(in2) ? 32'b1 : 32'b0;  // slti
            4'b0111: result = in1 + {in2, 12'b0};                           // auipc
            4'b1000: result = in2;                                          // lui
            default: result = 32'b0;
        endcase
    end

    assign zero = (result == 32'b0);
endmodule

module MulDiv(
    input clk,
    input rst_n,
    input valid,
    output reg ready,
    input  [1:0] control,
    input  [31:0] in_A,
    input  [31:0] in_B,
    output [31:0] out);

    localparam IDLE = 2'b00;
    localparam MUL  = 2'b01;
    localparam DIV  = 2'b10;

    reg [1:0] state, next_state;
    reg [5:0] calc_counter;

    reg [31:0] multiplicand;
    reg [64:0] product;

    reg [63:0] divisor;
    reg [64:0] remainder;

    // State machine
    always @(*) begin
        next_state = state;
        case(state)
            IDLE: begin
                if(valid) begin
                    case(control)
                        2'b00: next_state = MUL; // mul
                        2'b01,
                        2'b10: next_state = DIV; // divu or remu
                        default: next_state = IDLE;
                    endcase
                end else begin
                    next_state = IDLE;
                end
            end
            MUL: begin
                if(calc_counter == 6'd31)
                    next_state = IDLE;
                else
                    next_state = MUL;
            end
            DIV: begin
                if(calc_counter == 6'd31)
                    next_state = IDLE;
                else
                    next_state = DIV;
            end
            default: next_state = IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // Counter
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            calc_counter <= 6'd0;
        end else begin
            if((state == MUL || state == DIV) && next_state != IDLE)
                calc_counter <= calc_counter + 1;
            else
                calc_counter <= 6'd0;
        end
    end

    // Ready
    always @(*) begin
        ready = 1'b0;
        if((state == MUL || state == DIV) && calc_counter == 6'd31)
            ready = 1'b1;
    end

    // Output
    assign out = ( (state == MUL) && (calc_counter == 6'd31) ) ? product[31:0] :
             ( (state == DIV) && (calc_counter == 6'd31) && (control == 2'b01) ) ? remainder[31:0] :
             ( (state == DIV) && (calc_counter == 6'd31) && (control == 2'b10) ) ? remainder[64:33] :
             32'b0;

    // MulDiv
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            multiplicand <= 32'b0;
            product <= 65'b0;
            divisor <= 64'b0;
            remainder <= 65'b0;
        end else begin
            case(state)
                IDLE: begin
                    if(valid) begin
                        case(control)
                            2'b00: begin // mul
                                multiplicand <= in_B;
                                if (in_A[0] == 1) begin
                                    product <= ({33'b0, in_A} + {1'b0, in_B, 32'b0}) >> 1;
                                end else begin
                                    product <= {33'b0, in_A} >> 1;
                                end
                            end
                            2'b01, 2'b10: begin // divu remu
                                divisor <= {in_B, 32'b0};
                                if ({33'b0, in_A} >= {in_B, 32'b0}) begin
                                    remainder <= ((({33'b0, in_A} << 1) - {in_B, 32'b0}) << 1) + 1;
                                end else begin
                                    remainder <= {33'b0, in_A} << 2;
                                end
                            end
                        endcase
                    end
                end

                MUL: begin
                    if (product[0] == 1)
                        product <= (product + {1'b0, multiplicand, 32'b0}) >> 1;
                    else
                        product <= product >> 1;
                end

                DIV: begin
                    if (remainder >= divisor)
                        remainder <= ((remainder - divisor) << 1) + 1;
                    else
                        remainder <= remainder << 1;
                end
            endcase
        end
    end
endmodule
