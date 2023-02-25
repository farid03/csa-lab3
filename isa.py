# pylint :  disable=invalid-name
# pylint :  disable=consider-using-f-string
# pylint :  disable=missing-function-docstring
# pylint :  disable=missing-class-docstring


"""
isa
"""
import json
from enum import Enum
from typing import Tuple


class Opcode(str, Enum):
    """
    Opcode
    """
    # // относительная адресация
    LW = 'LW'  # A <- [B]
    SW = 'SW'  # [A] <- B

    JMP = 'JMP'  # unconditional transition
    # a,b,i
    BEQ = "BEQ"  # Branch if Equal (A == B)
    BNE = "BNE"  # Branch if Not Equal (A != B)
    BLT = "BLT"  # Branch if Less than (A < B)
    BGT = "BGT"  # Branch if Greater than (A > B)
    BNL = "BNL"  # Branch if Not Less than (A >= B)
    BNG = "BNG"  # Branch if less or equals then (A <= B)

    IN = "IN"  # Инструкции для обеспечения  IO по порту (в качестве аргумента принимается только номер порта)
    OUT = "OUT"

    ADD = 'ADD'  # t,a,b
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    REM = "REM"

    HALT = 'HALT'


ops_gr = {
    "mem": {
        Opcode.LW,
        Opcode.SW,
    },
    "branch": {
        Opcode.JMP,
        Opcode.BEQ,
        Opcode.BNE,
        Opcode.BLT,
        Opcode.BNL,
        Opcode.BGT,
        Opcode.BNG,
    },
    "arith": {
        Opcode.ADD,
        Opcode.SUB,
        Opcode.MUL,
        Opcode.DIV,
        Opcode.REM,
    },
    "io": {
        Opcode.IN,
        Opcode.OUT,
    }
}

registerToNumber = {
    "r0":  0x0,
    "r1":  0x1,
    "r2":  0x2,
    "r3":  0x3,
    "r4":  0x4,
    "r5":  0x5,
}

addrInstructionCode = {
    Opcode.ADD: 0x2,
    Opcode.SUB: 0x3,
    Opcode.MUL: 0x4,
    Opcode.DIV: 0x5,
    Opcode.REM: 0x6,
    Opcode.LW: 0x7,
    Opcode.SW: 0x8
}

branchInstructionCode = {
    Opcode.JMP: 0x0,
    Opcode.BEQ: 0x1,
    Opcode.BNE: 0x2,
    Opcode.BLT: 0x3,
    Opcode.BNL: 0x4,
    Opcode.BGT: 0x5,
    Opcode.BNG: 0x6,
}

ioInstructionCode = {
    Opcode.IN: 0x0,
    Opcode.OUT: 0x1,
}

STDIN_PORT, STDOUT_PORT = 0, 1


def write_code(filename: str, hex_data: list, hex_program: list):
    """Записать машинный код в файл."""
    file = open(filename, "wb")
    for instr in hex_program:  # записываем инструкции
        file.write(int(instr, 16).to_bytes(4, byteorder="big"))

    for i in hex_data:  # записываем данные
        file.write(int(i, 16).to_bytes(1, byteorder="big"))


def read_code(filename: str) -> bytes:
    """Прочесть машинный код из файла."""
    file = open(filename, "rb")

    return file.read()
