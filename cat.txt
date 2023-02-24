section data:
buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
aboba: 1
str: 'aaaa', 0

section text:
    _start:
        add r2,   [r0]  , buffer
        add r3,r0,6969 ; STDIN
    read:
        lw r1,r3
        sw r2,r1
        beq r1,r0,finish_read
        add r2,r2,1
        jmp read
    finish_read:
        add r2,r0,buffer
        add r3,r0,9696 ; STDOUT

    write:
        lw r1,r2
        sw r3,r1
        beq r1,r0,end ; f
        add r2,r2,1
        jmp write
    end:
        halt
