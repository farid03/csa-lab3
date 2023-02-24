#!/usr/bin/python3
# pylint: disable=missing-function-docstring  # чтобы не быть Капитаном Очевидностью
# pylint: disable=invalid-name                # сохраним традиционные наименования сигналов
# pylint: disable=consider-using-f-string     # избыточный синтаксис

import re
import sys

from isa import write_code, STDIN_PORT, STDOUT_PORT


def pre_process(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        line = line.partition(";")[0].strip()  # remove comments and spaces
        lines.append(line)

    text = " ".join(lines)
    # избавляется от лишних пробелов и символов перехода строки
    text = re.sub(" +", " ", text)

    return text


def tokenize(text):
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


def parse_data(tokens: list[str]):
    data = []
    labels = {}
    for token in tokens:
        if isinstance(token, tuple):
            labels[token[0]] = len(data)
        elif token.isdigit():
            data.append(token)

    return data, labels


def get_addressing_type(arg: str):
    if re.fullmatch('r\d{1,2}', arg):
        return 0
    if re.fullmatch('\[r\d{1,2}\]', arg):
        return 1
    # if arg.isdigit():
    return 2


def parse_instructions(tokens: list[str]):
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
            code.append({"opcode": statement, "addr_type": addr_type, "args": [tokens[i + 1].strip("[]"), tokens[i + 2]]})
            i += 3
            pass
        elif statement in ["JMP", "BEQ", "BNE", "BLT", "BGT", "BNL", "BNG"]:
            code.append({"opcode": statement, "args": [tokens[i + 1], tokens[i + 2], tokens[i + 3]]})
            i += 4
            pass
        elif statement in ["ADD", "SUB", "MUL", "DIV", "REM"]:
            addr_type = get_addressing_type(tokens[i + 3])
            code.append(
                {"opcode": statement, "addr_type": addr_type, "args": [tokens[i + 1], tokens[i + 2].strip("[]"), tokens[i + 3]]})
            i += 4
            pass
        else:
            # i += 1
            raise SyntaxError(f"Неизвестная инструкция: {statement}")
# TODO добавить бинарное представление (написать методы преобразования в бинарный вид имеющихся функций)
    return code, labels


def translate_to_struct(text):
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


def translate_to_binary(data: list, program: list):
    return


def main(args):
    assert len(args) == 2, \
        "Wrong arguments: translator.py <input_file> <target_file>"

    source, target = args

    with open(source, "rt", encoding="utf-8") as f:
        source = f.read()

    data, program = translate_to_struct(source)

    print("source LoC:", len(source.split()), "code instr:",
          len(program))

    write_code(target, data, program)


if __name__ == '__main__':
    print(format(0x0012, 'X'))
    main(sys.argv[1:])
