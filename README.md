# ASM-like. Транслятор и модель

Вариант: 
```
asm | risc | neum | hw | instr | binary | stream | port | prob1
```

| Особенности             |                                                                     |
|-------------------------|---------------------------------------------------------------------|
| ЯП. Синтаксис           | Синтаксис ассемблера. Поддержка меток (labels).                     |
| Архитектура             | Система команд должна быть упрощенной, в духе RISC архитектур.      |
| Организация памяти      | Архитектура фон Неймана.                                            |
| Control Unit            | hardwired. Реализуется как часть модели.                            |
| Точность модели         | Процессор необходимо моделировать с точностью до каждой инструкции. |
| Представление маш. кода | Бинарное представление.                                             |
| Ввод-вывод              | Ввод-вывод осуществляется как поток токенов.                        |
| Ввод-вывод ISA          | Port-mapped                                                         |
| Алгоритм                | Сумма всех чисел кратных 3 или 5 ниже 1000.                         |

Точность модели -- запись в журнале логов после каждой инструкции
Бинарное представлени -- придумать как упаковывать машинные команды и написать сериализацию/десериализацию, генерировать вместе с кодом мнемоники
Ввод-вывод ISA -- ввод вывод с точки зрения команд

## Язык программирования

Описание синтаксиса:
```text
<program> ::= 
        "section data:" <whitespace>* <data_section>?
        <whitespace> 
        "section text:" <whitespace>* <instruction_section>?
<data_section> ::= <data> (<whitespace> <data>)*
<data> ::= (<label_declaration>) " "* (<char_literal> | <number>) ("," (<char_literal> | <number>))*
<instruction_section> ::= <instruction> (<whitespace> <instruction>)*
<instruction> ::= (<label_declaration>)? " "* <letter>+ (" " (<address>  | (<reg> "," <address>) | (<reg> "," <reg> "," <address>)))? 
<address> ::= <number> | <label>
<reg> ::= "x" <number>
<label_declaration> ::= <label> ":"
<label> ::= <letter> (<letter>|<digit>)*
<char_literal> ::= "'" (<letter> | <digit> | <whitespace>)+ "'"
<letter> ::= "A" | "B" | "C" | "D" | "E" | "F" | "G"
       | "H" | "I" | "J" | "K" | "L" | "M" | "N"
       | "O" | "P" | "Q" | "R" | "S" | "T" | "U"
       | "V" | "W" | "X" | "Y" | "Z" | "a" | "b"
       | "c" | "d" | "e" | "f" | "g" | "h" | "i"
       | "j" | "k" | "l" | "m" | "n" | "o" | "p"
       | "q" | "r" | "s" | "t" | "u" | "v" | "w"
       | "x" | "y" | "z"
<whitespace> ::= " " | "\n" | "\t"
<number> ::= <digit> | ( [1-9] <digit>+ )
<digit> ::= [0-9]

```