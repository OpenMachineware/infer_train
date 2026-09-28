	.build_version macos, 27, 0	sdk_version 27, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_test_attention                 ; -- Begin function test_attention
	.p2align	2
_test_attention:                        ; @test_attention
	.cfi_startproc
; %bb.0:
	cmp	w4, #1
	b.lt	LBB0_32
; %bb.1:
	sub	sp, sp, #208
	stp	d11, d10, [sp, #80]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #96]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #112]            ; 16-byte Folded Spill
	stp	x26, x25, [sp, #128]            ; 16-byte Folded Spill
	stp	x24, x23, [sp, #144]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #160]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #176]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #192]            ; 16-byte Folded Spill
	add	x29, sp, #192
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	.cfi_offset b10, -120
	.cfi_offset b11, -128
	mov	x19, x7
	mov	x21, x5
	sxtw	x25, w5
	sbfiz	x8, x5, #1, #32
	str	x8, [sp, #24]                   ; 8-byte Folded Spill
	mov	w26, w5
	cmp	w6, #1
	b.lt	LBB0_24
; %bb.2:
	mov.16b	v8, v0
	mov	x27, x2
	mov	x22, x1
	mov	x24, x0
	mov	x10, #0                         ; =0x0
	mov	w8, w4
	mov	w28, w6
	lsl	x11, x25, #1
	stp	x11, x8, [sp, #8]               ; 16-byte Folded Spill
	mul	x8, x25, x8
	lsl	x20, x8, #1
	add	x8, x3, #16
	str	x8, [sp, #64]                   ; 8-byte Folded Spill
	lsl	x8, x25, #2
	str	x8, [sp]                        ; 8-byte Folded Spill
	fmov	s9, #1.00000000
	b	LBB0_4
LBB0_3:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldr	x10, [sp, #32]                  ; 8-byte Folded Reload
	add	x10, x10, #1
	ldp	x22, x27, [sp, #48]             ; 16-byte Folded Reload
	ldp	x11, x8, [sp]                   ; 16-byte Folded Reload
	add	x22, x22, x8
	add	x24, x24, x8
	add	x27, x27, x8
	ldr	x8, [sp, #64]                   ; 8-byte Folded Reload
	add	x8, x8, x11
	str	x8, [sp, #64]                   ; 8-byte Folded Spill
	ldr	x8, [sp, #16]                   ; 8-byte Folded Reload
	cmp	x10, x8
	b.eq	LBB0_31
LBB0_4:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
                                        ;       Child Loop BB0_14 Depth 3
                                        ;       Child Loop BB0_19 Depth 3
                                        ;     Child Loop BB0_23 Depth 2
	stp	x10, x24, [sp, #32]             ; 16-byte Folded Spill
	mov	x0, x19
	ldr	x1, [sp, #24]                   ; 8-byte Folded Reload
	bl	_bzero
	mov	x23, #0                         ; =0x0
	movi.2d	v10, #0000000000000000
	mov	w8, #-8388608                   ; =0xff800000
	fmov	s0, w8
	stp	x22, x27, [sp, #48]             ; 16-byte Folded Spill
	ldr	x24, [sp, #40]                  ; 8-byte Folded Reload
	b	LBB0_8
LBB0_5:                                 ;   in Loop: Header=BB0_8 Depth=2
	fadd	s10, s1, s9
LBB0_6:                                 ;   in Loop: Header=BB0_8 Depth=2
	add	x23, x23, #1
	cmp	x23, x28
	b.eq	LBB0_21
LBB0_7:                                 ;   in Loop: Header=BB0_8 Depth=2
	add	x22, x22, x20
	add	x27, x27, x20
	mov.16b	v0, v11
LBB0_8:                                 ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_10 Depth 3
                                        ;       Child Loop BB0_14 Depth 3
                                        ;       Child Loop BB0_19 Depth 3
	cmp	w21, #1
	b.lt	LBB0_16
; %bb.9:                                ;   in Loop: Header=BB0_8 Depth=2
	mov	x8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	mov	x9, x24
	mov	x10, x22
LBB0_10:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q2, [x9], #16
	ldr	q3, [x10], #16
	fcvtl	v4.4s, v2.4h
	fcvtl	v5.4s, v3.4h
	fmul.4s	v4, v4, v5
	fcvtl2	v2.4s, v2.8h
	fcvtl2	v3.4s, v3.8h
	fmul.4s	v2, v2, v3
	fadd.4s	v2, v4, v2
	fadd.4s	v1, v1, v2
	add	x8, x8, #8
	cmp	x8, x26
	b.lo	LBB0_10
; %bb.11:                               ;   in Loop: Header=BB0_8 Depth=2
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	fmul	s1, s8, s1
	fcmp	s1, s0
	fcsel	s11, s1, s0, gt
	b.le	LBB0_17
LBB0_12:                                ;   in Loop: Header=BB0_8 Depth=2
	fsub	s0, s0, s1
	bl	_expf
	fmul	s1, s10, s0
	cmp	w21, #0
	b.le	LBB0_5
; %bb.13:                               ;   in Loop: Header=BB0_8 Depth=2
	mov	x8, #0                          ; =0x0
	fcvt	h0, s0
	mov	x9, x19
LBB0_14:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q2, [x9]
	fmul.8h	v2, v2, v0[0]
	str	q2, [x9], #16
	add	x8, x8, #8
	cmp	x8, x26
	b.lo	LBB0_14
; %bb.15:                               ;   in Loop: Header=BB0_8 Depth=2
	fmov	s0, #1.00000000
	fadd	s10, s1, s0
	b	LBB0_18
LBB0_16:                                ;   in Loop: Header=BB0_8 Depth=2
	movi.2d	v1, #0000000000000000
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	fmul	s1, s8, s1
	fcmp	s1, s0
	fcsel	s11, s1, s0, gt
	b.gt	LBB0_12
LBB0_17:                                ;   in Loop: Header=BB0_8 Depth=2
	fsub	s0, s1, s0
	bl	_expf
	fadd	s10, s10, s0
	cmp	w21, #0
	b.le	LBB0_6
LBB0_18:                                ;   in Loop: Header=BB0_8 Depth=2
	mov	x8, #0                          ; =0x0
	fcvt	h0, s0
	mov	x9, x19
	mov	x10, x27
LBB0_19:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q1, [x10], #16
	ldr	q2, [x9]
	fmla.8h	v2, v1, v0[0]
	str	q2, [x9], #16
	add	x8, x8, #8
	cmp	x8, x26
	b.lo	LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_8 Depth=2
	add	x23, x23, #1
	cmp	x23, x28
	b.ne	LBB0_7
	b	LBB0_22
LBB0_21:                                ;   in Loop: Header=BB0_4 Depth=1
	cmp	w21, #1
	b.lt	LBB0_3
LBB0_22:                                ;   in Loop: Header=BB0_4 Depth=1
	mov	x8, #0                          ; =0x0
	ldr	x9, [sp, #64]                   ; 8-byte Folded Reload
	mov	x10, x19
	fdiv	s0, s9, s10
LBB0_23:                                ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q1, [x10], #16
	fcvtl	v2.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v2, v2, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q2, q1, [x9, #-16]
	add	x8, x8, #8
	add	x9, x9, #32
	cmp	x8, x25
	b.lt	LBB0_23
	b	LBB0_3
LBB0_24:
	cmp	w21, #0
	b.le	LBB0_29
; %bb.25:
	mov	x20, #0                         ; =0x0
	mov	w21, w4
	add	x22, x3, #16
	lsl	x23, x26, #2
	mvni.4s	v0, #127, msl #16
	fneg.4s	v2, v0
	str	q2, [sp, #64]                   ; 16-byte Folded Spill
LBB0_26:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_27 Depth 2
	mov	x0, x19
	ldr	x1, [sp, #24]                   ; 8-byte Folded Reload
	bl	_bzero
	ldr	q2, [sp, #64]                   ; 16-byte Folded Reload
	mov	x8, #0                          ; =0x0
	mov	x9, x22
	mov	x10, x19
LBB0_27:                                ;   Parent Loop BB0_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q0, [x10], #16
	fcvtl	v1.4s, v0.4h
	fcvtl2	v0.4s, v0.8h
	fmul.4s	v1, v1, v2
	fmul.4s	v0, v0, v2
	stp	q1, q0, [x9, #-16]
	add	x8, x8, #8
	add	x9, x9, #32
	cmp	x8, x25
	b.lt	LBB0_27
; %bb.28:                               ;   in Loop: Header=BB0_26 Depth=1
	add	x20, x20, #1
	add	x22, x22, x23
	cmp	x20, x21
	b.ne	LBB0_26
	b	LBB0_31
LBB0_29:
	mov	x0, x19
	ldr	x1, [sp, #24]                   ; 8-byte Folded Reload
	mov	x19, x4
	bl	_bzero
	mov	x8, x19
LBB0_30:                                ; =>This Inner Loop Header: Depth=1
	subs	w8, w8, #1
	b.ne	LBB0_30
LBB0_31:
	ldp	x29, x30, [sp, #192]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #176]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #160]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #144]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #128]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #112]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #96]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #80]             ; 16-byte Folded Reload
	add	sp, sp, #208
LBB0_32:
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
