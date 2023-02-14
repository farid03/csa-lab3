#!/usr/bin/python3
# pylint: disable=missing-function-docstring  # чтобы не быть Капитаном Очевидностью
# pylint: disable=invalid-name                # сохраним традиционные наименования сигналов
# pylint: disable=consider-using-f-string     # избыточный синтаксис

"""Транслятор brainfuck в машинный код
"""

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


def allocate(tokens):
    data = []
    labels = {}
    for token in tokens:
        if isinstance(token, tuple):
            labels[token[0]] = len(data)
        elif token.isdigit():
            data.append(token)

    return data, labels


def parse(tokens):
    labels = {}
    code = []
    args_count = 0
    for token in tokens:
        if isinstance(token, tuple):
            labels[token[0]] = len(code)
        else:
            token_upper = token.upper()
            if args_count > 0:
                if args_count != 0 and token[0] == 'x' and token[1:].isdigit():
                    token = token[1:]
                code[-1]["args"].append(token)
                args_count -= 1
            else:
                code.append({"opcode": token_upper, "args": []})
                if token_upper == 'HALT':
                    args_count = 0
                elif token_upper in ["IN", "OUT", "JMP"]:
                    args_count = 1
                elif token_upper in ["SW", "LW"]:
                    args_count = 2
                else:
                    args_count = 3
    return code, labels


def translate(text):
    text = pre_process(text)
    # data_tokens - переменные, text_tokens - инструкции
    data_tokens, text_tokens = tokenize(text)
    data, data_labels = allocate(data_tokens)
    code, code_labels = parse(text_tokens)

    program = code
    for word_idx, word in enumerate(program):
        if isinstance(word, dict):
            for arg_idx, arg in enumerate(word["args"]):
                if arg in data_labels:
                    program[word_idx]["args"][arg_idx] = data_labels[arg]
                elif arg in code_labels:
                    program[word_idx]["args"][arg_idx] = code_labels[arg]
    return data, program


def main(args):
    assert len(args) == 2, \
        "Wrong arguments: translator.py <input_file> <target_file>"

    source, target = args

    with open(source, "rt", encoding="utf-8") as f:
        source = f.read()

    data, program = translate(source)

    print("source LoC:", len(source.split()), "code instr:",
          len(program))

    write_code(target, data, program)


if __name__ == '__main__':
    main(sys.argv[1:])
