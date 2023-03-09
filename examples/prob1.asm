section data:
sum: 0
current_number: 1001
result: 1

section text:
    _start:
        lw r1,current_number
    loop:
        sub r1,r1,1

        beq r1,r0,end

        rem r3,r1,3
        bne r0,r3,next
        add r4,r4,r1

        jmp loop
    next:
        rem r3,r1,5
        bne r0,r3,loop
        add r4,r4,r1

        jmp loop

    end:
        add r1,r0,r0
        add r1,r0,sum
        sw r1,r4
        halt