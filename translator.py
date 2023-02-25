#!/usr/bin/python3
# pylint: disable=missing-function-docstring  # чтобы не быть Капитаном Очевидностью
# pylint: disable=invalid-name                # сохраним традиционные наименования сигналов
# pylint: disable=consider-using-f-string     # избыточный синтаксис

import re
import sys

from typing import Tuple
from isa import write_code, STDIN_PORT, STDOUT_PORT, \
    ops_gr, addr_instruction_code, branch_instruction_code, io_instruction_code, register_to_number, Opcode


def pre_process(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        line = line.partition(";")[0].strip()  # remove comments and spaces
        lines.append(line)

    text = " ".join(lines)
    # избавляется от лишних пробелов и символов перехода строки
    text = re.sub(" +", " ", text)

    return text


def tokenize(text) -> Tuple[list, list]:
    text = re.sub(
        r"'.*'",
        lambda match: f'{",".join(map(lambda char: str(ord(char)), match.group()[1:-1]))}',
        text
    )
    data_section_index = text.find("section data:")
    text_section_index = text.find("section text:")

    data_tokens = re.split(
        "[, ]", text[data_section_index + len("section data:"): text_section_index])
    data_tokens = list(filter(lambda token: token, data_tokens))
    data_tokens = list(
        map(lambda token: (token[:-1],) if token[-1] == ':' else token, data_tokens))

    text_tokens = re.split(
        "[, ]", text[text_section_index + len("section text:"):])
    text_tokens = list(filter(lambda token: token, text_tokens))
    text_tokens = list(
        map(lambda token: (token[:-1],) if token[-1] == ':' else token, text_tokens))
    return data_tokens, text_tokens


def parse_data(tokens: list[str]) -> Tuple[list, dict]:
    data = []
    labels = {}
    for token in tokens:
        if isinstance(token, tuple):
            labels[token[0]] = len(data)
        elif token.isdigit():
            data.append(token)

    return data, labels


def get_addressing_type(arg: str) -> int:
    if re.fullmatch('r\d{1,2}', arg):
        return 0
    if re.fullmatch('\[r\d{1,2}\]', arg):
        return 1
    # if arg.isdigit():
    return 2


def parse_instructions(tokens: list[str]) -> Tuple[list, dict]:
    labels = {}
    code = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if isinstance(token, tuple):
            labels[token[0]] = len(code)
            i += 1
            continue

        statement = tokens[i].upper()
        if statement in ["HALT"]:
            code.append({"opcode": statement, "args": []})
            i += 1
            pass
        elif statement in ["JMP"]:
            code.append({"opcode": statement, "args": [tokens[i + 1]]})
            i += 2
            pass
        elif statement in ["IN", "OUT"]:
            code.append({"opcode": statement, "args": [tokens[i + 1], tokens[i + 2]]})
            i += 3
            pass
        elif statement in ["SW", "LW"]:
            addr_type = get_addressing_type(tokens[i + 2])
            code.append(
                {"opcode": statement, "addr_type": addr_type, "args": [tokens[i + 1].strip("[]"), tokens[i + 2]]})
            i += 3
            pass
        elif statement in ["JMP", "BEQ", "BNE", "BLT", "BGT", "BNL", "BNG"]:
            code.append({"opcode": statement, "args": [tokens[i + 1], tokens[i + 2], tokens[i + 3]]})
            i += 4
            pass
        elif statement in ["ADD", "SUB", "MUL", "DIV", "REM"]:
            addr_type = get_addressing_type(tokens[i + 3])
            code.append(
                {"opcode": statement, "addr_type": addr_type,
                 "args": [tokens[i + 1], tokens[i + 2].strip("[]"), tokens[i + 3]]})
            i += 4
            pass
        else:
            raise SyntaxError(f"Неизвестная инструкция: {statement}")

    return code, labels


def translate_to_struct(text) -> Tuple[list, list[dict]]:
    text = pre_process(text)
    # data_tokens - переменные, text_tokens - инструкции
    data_tokens, text_tokens = tokenize(text)
    data, data_labels = parse_data(data_tokens)
    code, code_labels = parse_instructions(text_tokens)

    # резолвим метки и поля
    program = code
    for word_idx, word in enumerate(program):
        if isinstance(word, dict):
            for arg_idx, arg in enumerate(word["args"]):
                if arg in data_labels:
                    program[word_idx]["args"][arg_idx] = data_labels[arg] + len(code) * 4
                elif arg in code_labels:
                    program[word_idx]["args"][arg_idx] = code_labels[arg] * 4

    return data, program


def translate_to_binary(data: list, program: list[dict]) -> Tuple[list[str], list[str]]:
    hex_program = list()
    for instr in program:
        hex_instr = translate_instruction_to_hex_str(instr)
        print(f"{hex_instr} {instr}")
        hex_program.append(hex_instr)

    hex_data_bytes = list()
    for i in data:
        hex_data_bytes.append(to_bytes_str(int(i), 2))

    return hex_data_bytes, hex_program


def translate_instruction_to_hex_str(instr: dict) -> str:
    hex_instr = ""  # hex-предстваление 32битной инструкции (в виде строки)
    _instr_type = ""  # для вывода ошибок
    opcode = Opcode(instr["opcode"])

    if opcode in ops_gr["arith"]:  # addr instruction
        _instr_type = "arith"
        hex_instr = "{0}{1}{2}{3}".format(
            get_lower_nibble(addr_instruction_code[opcode]),
            get_lower_nibble(int(instr["addr_type"])),
            get_lower_nibble(register_to_number[instr["args"][0]]),
            get_lower_nibble(register_to_number[instr["args"][1]]))
        if int(instr["addr_type"]) == 2:
            hex_instr += to_bytes_str(int(instr["args"][2]), 4)
        else:
            hex_instr += get_lower_nibble(register_to_number[instr["args"][2]])
            hex_instr += "0" * 3

    elif opcode in ops_gr["mem"]:  # addr instruction
        _instr_type = "mem"
        hex_instr = "{0}{1}{2}".format(
            get_lower_nibble(addr_instruction_code[opcode]),
            get_lower_nibble(int(instr["addr_type"])),
            get_lower_nibble(register_to_number[instr["args"][0]]))
        if int(instr["addr_type"]) == 2:
            hex_instr += "0" \
                         + to_bytes_str(instr["args"][1], 4)
        else:
            hex_instr += get_lower_nibble(register_to_number[instr["args"][1]]) \
                         + "0" * 4

    elif opcode is Opcode.HALT:
        _instr_type = "halt"
        hex_instr = "00000010"

    elif opcode in ops_gr["branch"]:
        _instr_type = "branch"
        hex_instr = "{0}{1}".format(
            get_lower_nibble(int("1111", 2)),
            get_lower_nibble(branch_instruction_code[opcode]))
        if opcode is Opcode.JMP:
            hex_instr += "00{0}".format(to_bytes_str(int(instr["args"][0]), 4))
        else:
            hex_instr += "{0}{1}{2}".format(
                get_lower_nibble(register_to_number[instr["args"][0]]),
                get_lower_nibble(register_to_number[instr["args"][1]]),
                to_bytes_str(int(instr["args"][2]), 4))

    elif opcode in ops_gr["io"]:
        _instr_type = "io"
        hex_instr = "{0}{1}{2}0{3}".format(
            get_lower_nibble(int("0001", 2)),
            get_lower_nibble(io_instruction_code[opcode]),
            get_lower_nibble(register_to_number[instr["args"][0]]),
            to_bytes_str(int(instr["args"][1]), 4))

    assert len(hex_instr) == 8, f"Error in translate {_instr_type}-instruction to binary: {str(instr)},\n " \
                                f"result: {hex_instr}"

    return hex_instr


def get_lower_nibble(byte: int) -> str:
    return to_bytes_str(byte, 1)


def to_bytes_str(number: int, len_in_nibbles: int) -> str:
    hex_num = hex(number).replace("0x", "")

    if len(hex_num) >= len_in_nibbles:
        return hex_num[len(hex_num) - len_in_nibbles:]

    return (len_in_nibbles - len(hex_num)) * "0" + hex_num


def main(args):
    assert len(args) == 2, \
        "Wrong arguments: translator.py <input_file> <target_file>"

    source, target = args

    with open(source, "rt", encoding="utf-8") as f:
        source = f.read()

    data, program = translate_to_struct(source)
    hex_data, hex_program = translate_to_binary(data, program)

    print("source LoC:", len(source.split()), "code instr:",
          len(program))

    write_code(target, hex_data, hex_program)


if __name__ == '__main__':
    main(sys.argv[1:])
