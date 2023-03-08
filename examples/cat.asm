section data:
buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

section text:
    _start:
        add r2,r0,buffer
    read:
        in r1, 0
        sw r2,r1
        beq r1,r0,finish_read ; пока не достигнем нуль-терминатора
        add r2,r2,1
        jmp read
    finish_read:
        add r2,r0,buffer

    write:
        lw r1,r2
        out r1, 1
        beq r1,r0,end
        add r2,r2,1
        jmp write
    end:
        halt