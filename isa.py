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

STDIN_PORT, STDOUT_PORT = 0, 1


def write_code(filename: str, data: list, program: list):
    """Записать машинный код в файл."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(json.dumps({"data": data, "code": program}, indent=4))


def read_code(filename: str) -> Tuple[list, list[dict]]:
    """Прочесть машинный код из файла."""
    with open(filename, encoding="utf-8") as file:
        content = json.loads(file.read())
        return content["data"], content["code"]
