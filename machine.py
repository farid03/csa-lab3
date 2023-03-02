#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string

import logging
from collections import deque
import sys

from isa import Opcode, read_code, STDOUT_PORT, STDIN_PORT, ops_gr


class RegisterUnit:
    registers: list[int]
    rd: int
    rs1: int
    rs2: int

    def __init__(self, registers_count: int, stack_vertex: int) -> None:
        self.registers = [0] * registers_count
        self.registers[registers_count - 1] = stack_vertex
        self.rd = 0
        self.rs1 = 0
        self.rs2 = 0

    def latch_sel_rd(self, number: int):
        self.rd = number

    def latch_sel_rs1(self, number: int):
        self.rs1 = number

    def latch_sel_rs2(self, number: int):
        self.rs2 = number

    def get_rs1_data(self):
        return self.registers[self.rs1]

    def get_rs2_data(self):
        return self.registers[self.rs2]

    def set_rd_data(self, data):
        if self.rd != 0:
            self.registers[self.rd] = int(data)


class ALU:
    output: int
    a: int
    b: int
    _operations_ = {
        Opcode.ADD: lambda a, b: a + b,
        Opcode.SUB: lambda a, b: a - b,
        Opcode.MUL: lambda a, b: a * b,
        Opcode.DIV: lambda a, b: a // b,
        Opcode.REM: lambda a, b: a % b,
    }

    def __init__(self) -> None:
        self.output = 0
        self.a = 0
        self.b = 0

    def load(self, a: int, b: int):
        self.a = a
        self.b = b

    def compute(self, operation: Opcode) -> int:
        self.output = int(self._operations_[operation](self.a, self.b))
        return self.output


class BranchComparator:
    a: int
    b: int

    def __init__(self) -> None:
        self.a = 0
        self.b = 0

    def load(self, a: int, b: int):
        self.a = a
        self.b = b

    def compare(self) -> tuple[bool, bool]:
        return self.a == self.b,  self.a < self.b


class IO:
    input_buffer: deque

    def __init__(self, input_tokens: list) -> None:
        self.input_buffer = deque(input_tokens)
        self.output_buffer = deque()

    def eof(self):
        return not self.input_buffer

    def input(self):
        return self.input_buffer.popleft()

    def output(self, character):
        self.output_buffer.append(character)


class DataPath:
    memory: list[dict | str]
    program_counter: int
    data_address: int
    data_memory_size: int
    memory: list[int]
    ru: RegisterUnit
    alu: ALU
    bc: BranchComparator
    io: IO

    immediately_generator: int
    current_instruction: dict
    args: deque[int]

    def __init__(self, program: list, data, data_memory_size: int, input_buffer: list):
        assert data_memory_size > 0, "Data_memory size should be non-zero"
        self.data_memory_size = data_memory_size
        self.program = program
        self.data_memory = data + ([0] * (data_memory_size - len(data)))
        self.data_address = 0
        self.program_counter = 0
        self.io = IO([ord(token) for token in input_buffer])
        self.immediately_generator = 0
        self.current_instruction = Opcode("HALT")
        self.args: deque[str]
        self.current_data = 0
        self.ru = RegisterUnit(5, stack_vertex=data_memory_size - 1)
        self.alu = ALU()
        self.bc = BranchComparator()

    def select_instruction(self) -> Opcode:
        self.current_instruction = self.program[self.program_counter]
        self.args = deque(self.current_instruction['args'])
        self.program_counter += 1

        return self.current_instruction['opcode']

    def latch_dest_reg_from_instr(self):
        arg = self.args.popleft()
        self.ru.latch_sel_rd(int(arg))

    def latch_rs1_from_instr(self):
        arg = self.args.popleft()
        self.ru.latch_sel_rs1(int(arg))

    def latch_rs2_from_instr(self):
        arg = self.args.popleft()
        self.ru.latch_sel_rs2(int(arg))

    def latch_imm_gen(self):
        self.immediately_generator = int(self.args.popleft())

    def latch_rs1_to_alu(self):
        self.alu.a = self.ru.get_rs1_data()

    def latch_rs2_to_alu(self):
        self.alu.b = self.ru.get_rs2_data()

    def latch_imm_to_alu(self):
        """Загружает непосредственное значение в ALU"""
        self.alu.b = self.immediately_generator

    def compute_ALU(self, opcode: Opcode):
        self.alu.compute(opcode)

    def latch_address_to_memory(self):
        """Загружает целевой адрес в память"""

        if self.ru.get_rs1_data() == STDIN_PORT:   # FIXME
            if self.io.eof():
                raise EOFError
            self.current_data = self.io.input()
        else:
            self.data_address = self.ru.get_rs1_data()
            self.current_data = self.data_memory[self.data_address]

    def store_data_to_memory_from_reg(self):
        """Загружает данные в память"""
        if self.ru.get_rs1_data() == STDOUT_PORT:  # FIXME
            self.io.output(chr(self.ru.get_rs2_data()))
        else:
            self.data_memory[self.ru.get_rs1_data()
                             ] = self.ru.get_rs2_data()

    def store_data_to_memory_from_imm(self):
        """Загружает данные в память"""
        if self.ru.get_rs1_data() == STDOUT_PORT:  # FIXME
            self.io.output(chr(self.immediately_generator))
        else:
            self.memory[self.ru.get_rs1_data(
            )] = self.immediately_generator

    def latch_address_to_memory_from_imm(self):
        if self.immediately_generator == STDIN_PORT:  # FIXME
            if self.io.eof():
                raise EOFError
            self.current_data = self.io.input()
        else:
            self.data_address = self.immediately_generator
            self.current_data = self.data_memory[self.data_address]

    def latch_reg_from_memory(self):
        """Значение из памяти перезаписывает регистр"""
        self.ru.set_rd_data(self.current_data)

    def latch_reg_from_alu(self):
        """ALU перезаписывает регистр"""
        self.ru.set_rd_data(self.alu.output)

    def latch_program_counter(self):
        """Перезаписывает значение PC из ImmGen"""
        self.program_counter = self.immediately_generator

    def latch_regs_to_bc(self):
        """Загружает регистры в Branch Comparator."""
        self.bc.a, self.bc.b =\
            self.ru.get_rs1_data(), self.ru.get_rs2_data()
        return self.bc.compare()

    def latch_instruct(self):
        opcode = Opcode(self.current_instruction["opcode"])
        if opcode is Opcode.JMP:
            self.latch_imm_gen()
        elif opcode is Opcode.LWI:
            self.latch_dest_reg_from_instr()
            self.latch_imm_gen()
        elif opcode is Opcode.LW:
            self.latch_dest_reg_from_instr()
            self.latch_rs1_from_instr()
        elif opcode is Opcode.SW:
            self.latch_rs1_from_instr()
            self.latch_rs2_from_instr()
        elif opcode is Opcode.SWI:
            self.latch_rs1_from_instr()
            self.latch_imm_gen()
        elif opcode in ops_gr["branch"]:
            self.latch_rs1_from_instr()
            self.latch_rs2_from_instr()
            self.latch_imm_gen()
        elif opcode in ops_gr['arith']:
            self.latch_dest_reg_from_instr()
            self.latch_rs1_from_instr()
            if opcode in ops_gr["imm"]:
                self.latch_imm_gen()
            else:
                self.latch_rs2_from_instr()


class ControlUnit:
    data_path: DataPath

    def __init__(self, data_path):
        self.data_path = data_path
        self._tick = 0

    def tick(self):
        """Счётчик тактов процессора. Вызывается при переходе на следующий такт."""
        logging.debug('%s', self)
        self._tick += 1

    def current_tick(self):
        """Возвращает текущий такт."""
        return self._tick

    def decode_and_execute_instruction(self):
        # TODO добавить бинарное представление
        opcode = Opcode(self.data_path.select_instruction())
        dp = self.data_path
        dp.latch_instruct()
        self.tick()

        if opcode is Opcode.JMP:
            dp.latch_program_counter()
        elif opcode in ops_gr["branch"]:
            equals, less = dp.latch_regs_to_bc()
            self.tick()
            if any([
                opcode is Opcode.BEQ and equals,
                opcode is Opcode.BNE and not equals,
                opcode is Opcode.BLT and less,
                opcode is Opcode.BNL and not less,
                opcode is Opcode.BGT and not less and not equals,
                opcode is Opcode.BNG and (less or equals)
            ]):
                dp.latch_program_counter()
        elif opcode is Opcode.LWI:
            dp.latch_address_to_memory_from_imm()
            self.tick()
            dp.latch_reg_from_memory()
        elif opcode is Opcode.LW:
            dp.latch_address_to_memory()
            self.tick()
            dp.latch_reg_from_memory()
        elif opcode is Opcode.SW:
            dp.store_data_to_memory_from_reg()
        elif opcode is Opcode.SWI:
            dp.store_data_to_memory_from_imm()
        elif opcode in ops_gr["arith"]:
            if opcode in ops_gr["imm"]:
                dp.latch_imm_to_alu()
            else:
                dp.latch_rs2_to_alu()
            dp.latch_rs1_to_alu()
            dp.compute_ALU(opcode=opcode)
            self.tick()
            dp.latch_reg_from_alu()

        elif opcode is Opcode.HALT:
            raise StopIteration()

        self.tick()

    def __repr__(self):
        state = "{{TICK: {}, PC: {}, ADDR: {}, OUT: }}".format(
            self._tick,
            self.data_path.program_counter,
            self.data_path.data_address
            # self.data_path.output_buffer[0]
        )

        registers = "{{[rd: {}, rs1: {}, rs2: {}, imm: {}]  Regs {} }}".format(
            self.data_path.ru.rd,
            self.data_path.ru.rs1,
            self.data_path.ru.rs2,
            self.data_path.immediately_generator,
            f"[{' '.join([str(reg) for reg in self.data_path.ru.registers])}]"
        )

        opcode = self.data_path.current_instruction["opcode"]
        args = self.data_path.current_instruction['args']
        action = "{} {}".format(
            opcode, f"[{' '.join([str(arg) for arg in args])}]")
        alu = "ALU [a:{} b:{} output:{}]".format(
            self.data_path.alu.a, self.data_path.alu.b, self.data_path.alu.output)

        return "{} {} {} {} ".format(state, registers, alu, action)


def show_memory(data_memory) -> str:
    data_memory_state = ""

    for address, cell in enumerate(reversed(data_memory)):
        cell = int(cell)
        # binary representation == br
        address = len(data_memory) - address - 1
        cell_br = bin(cell)[2:]
        address_br = bin(address)[2:]
        cell_br = (32 - len(cell_br)) * "0" + cell_br
        address_br = (10 - len(address_br)) * "0" + address_br
        data_memory_state += f"[{{{address:5}}}\
    [{address_br:10}]  -> [{cell_br:32}] = ({cell:10})\n"
    return data_memory_state


def simulation(program: list[dict], data: list[str | int], input_tokens, data_memory_size, limit):
    """Запуск симуляции процессора.

    Длительность моделирования ограничена количеством выполненных инструкций.
    """
    logging.info("{ INPUT MESSAGE } [ `%s` ]", "".join(input_tokens))
    logging.info("{ INPUT TOKENS  } [ %s ]", ",".join(
        [str(ord(token)) for token in input_tokens]))

    data_path = DataPath(program, data, data_memory_size, input_tokens)
    control_unit = ControlUnit(data_path)
    instr_counter = 0

    try:
        while True:
            if not limit > instr_counter:
                print("too long execution, increase limit!")
                break
            control_unit.decode_and_execute_instruction()
            instr_counter += 1
    except EOFError:
        logging.warning('Input buffer is empty!')
    except StopIteration:
        pass

    return ''.join(data_path.io.output_buffer), instr_counter,\
        control_unit.current_tick(), show_memory(data_path.data_memory)


def main(args):

    assert len(args) == 2,\
        "Wrong arguments: machine.py <code.binary> <input>"
    code_file, input_file = args

    data, program = read_code(code_file)
    with open(input_file, encoding="utf-8") as file:
        input_text = file.read()
        input_token = []
        for char in input_text:
            input_token.append(char)
    input_token.append(chr(0))

    output, instr_counter, ticks, data_memory_state = "", "", "", ""
    output, instr_counter, ticks, data_memory_state = simulation(
        program, data,
        input_tokens=input_token,
        data_memory_size=125,
        limit=12000
    )
    logging.info("%s", f"Memory map is\n{data_memory_state}")

    print(f"Output is `{''.join(output)}`")
    print(f"instr_counter: {instr_counter} ticks: {ticks}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    main(sys.argv[1:])
