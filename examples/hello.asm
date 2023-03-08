section data:
hello: 'Hello, World!!!',0

section text:
    _start:
        add r2,r0,hello
    write:
        lw r1,r2
        beq r1,r0,end
        out r1, 1
        add r2,r2,1
        jmp write
    end:
        halt
