	.build_version macos, 27, 0	sdk_version 27, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_decode_attention_f16_native    ; -- Begin function decode_attention_f16_native
	.p2align	2
_decode_attention_f16_native:           ; @decode_attention_f16_native
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #352
	stp	d11, d10, [sp, #224]            ; 16-byte Folded Spill
	stp	d9, d8, [sp, #240]              ; 16-byte Folded Spill
	stp	x28, x27, [sp, #256]            ; 16-byte Folded Spill
	stp	x26, x25, [sp, #272]            ; 16-byte Folded Spill
	stp	x24, x23, [sp, #288]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #304]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #320]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #336]            ; 16-byte Folded Spill
	add	x29, sp, #336
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
	stp	x1, x2, [sp, #112]              ; 16-byte Folded Spill
	str	x0, [sp, #72]                   ; 8-byte Folded Spill
	cmp	w4, #1
	b.lt	LBB0_70
; %bb.1:
	mov	x19, x7
	mov	x21, x5
	sxtw	x24, w5
	sbfiz	x8, x5, #1, #32
	str	x8, [sp, #24]                   ; 8-byte Folded Spill
	mov	w26, w5
	cmp	w6, #1
	b.lt	LBB0_63
; %bb.2:
	mov.16b	v8, v0
	mov	x15, #0                         ; =0x0
	mov	x11, #0                         ; =0x0
	and	w8, w21, #0x7fffffe0
	str	w8, [sp, #140]                  ; 4-byte Folded Spill
	mov	w8, w4
	mov	w9, w6
	stp	x9, x8, [sp, #144]              ; 16-byte Folded Spill
	add	x9, x19, #2
	str	x9, [sp, #80]                   ; 8-byte Folded Spill
	mul	x8, x24, x8
	lsl	x8, x8, #1
	stur	x8, [x29, #-152]                ; 8-byte Folded Spill
	ldr	x20, [sp, #72]                  ; 8-byte Folded Reload
	add	x8, x20, #32
	str	x8, [sp, #128]                  ; 8-byte Folded Spill
	lsl	x8, x24, #1
	ldp	x27, x28, [sp, #112]            ; 16-byte Folded Reload
	add	x10, x27, #32
	add	x25, x28, #32
	add	x12, x19, #32
	str	x12, [sp, #104]                 ; 8-byte Folded Spill
	add	x12, x3, #16
	lsl	x13, x24, #2
	stp	x13, x8, [sp, #8]               ; 16-byte Folded Spill
	fmov	s9, #1.00000000
	str	x24, [sp, #160]                 ; 8-byte Folded Spill
	b	LBB0_4
LBB0_3:                                 ;   in Loop: Header=BB0_4 Depth=1
	add	x11, x11, #1
	ldr	x8, [sp, #128]                  ; 8-byte Folded Reload
	ldr	x9, [sp, #16]                   ; 8-byte Folded Reload
	add	x8, x8, x9
	str	x8, [sp, #128]                  ; 8-byte Folded Spill
	ldp	x25, x10, [sp, #56]             ; 16-byte Folded Reload
	add	x10, x10, x9
	add	x15, x15, x9
	ldp	x27, x28, [sp, #32]             ; 16-byte Folded Reload
	add	x27, x27, x9
	add	x20, x20, x9
	add	x25, x25, x9
	add	x28, x28, x9
	ldr	x8, [sp, #8]                    ; 8-byte Folded Reload
	add	x12, x12, x8
	ldr	x8, [sp, #152]                  ; 8-byte Folded Reload
	cmp	x11, x8
	b.eq	LBB0_70
LBB0_4:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_6 Depth 2
                                        ;       Child Loop BB0_9 Depth 3
                                        ;       Child Loop BB0_14 Depth 3
                                        ;       Child Loop BB0_21 Depth 3
                                        ;       Child Loop BB0_25 Depth 3
                                        ;       Child Loop BB0_29 Depth 3
                                        ;       Child Loop BB0_39 Depth 3
                                        ;       Child Loop BB0_42 Depth 3
                                        ;       Child Loop BB0_36 Depth 3
                                        ;       Child Loop BB0_53 Depth 3
                                        ;       Child Loop BB0_57 Depth 3
                                        ;       Child Loop BB0_51 Depth 3
                                        ;     Child Loop BB0_62 Depth 2
	str	x10, [sp, #64]                  ; 8-byte Folded Spill
	str	x12, [sp, #48]                  ; 8-byte Folded Spill
	str	x11, [sp, #168]                 ; 8-byte Folded Spill
	stur	x15, [x29, #-160]               ; 8-byte Folded Spill
	mul	x8, x11, x24
	ldr	x9, [sp, #72]                   ; 8-byte Folded Reload
	add	x23, x9, x8, lsl #1
	mov	x0, x19
	ldr	x22, [sp, #24]                  ; 8-byte Folded Reload
	mov	x1, x22
	bl	_bzero
	ldr	x11, [sp, #168]                 ; 8-byte Folded Reload
	ldur	x15, [x29, #-160]               ; 8-byte Folded Reload
	mov	x24, #0                         ; =0x0
	ldr	x8, [sp, #120]                  ; 8-byte Folded Reload
	madd	x8, x22, x11, x8
	str	x8, [sp, #96]                   ; 8-byte Folded Spill
	add	x8, x8, #2
	str	x8, [sp, #88]                   ; 8-byte Folded Spill
	mov	w8, #-8388608                   ; =0xff800000
	movi.2d	v10, #0000000000000000
	fmov	s11, w8
	stp	x27, x28, [sp, #32]             ; 16-byte Folded Spill
	str	x25, [sp, #56]                  ; 8-byte Folded Spill
	ldr	x22, [sp, #64]                  ; 8-byte Folded Reload
	b	LBB0_6
LBB0_5:                                 ;   in Loop: Header=BB0_6 Depth=2
	add	x24, x24, #1
	ldur	x8, [x29, #-152]                ; 8-byte Folded Reload
	add	x22, x22, x8
	add	x27, x27, x8
	add	x25, x25, x8
	add	x28, x28, x8
	ldr	x8, [sp, #144]                  ; 8-byte Folded Reload
	cmp	x24, x8
	ldr	x11, [sp, #168]                 ; 8-byte Folded Reload
	b.eq	LBB0_60
LBB0_6:                                 ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_9 Depth 3
                                        ;       Child Loop BB0_14 Depth 3
                                        ;       Child Loop BB0_21 Depth 3
                                        ;       Child Loop BB0_25 Depth 3
                                        ;       Child Loop BB0_29 Depth 3
                                        ;       Child Loop BB0_39 Depth 3
                                        ;       Child Loop BB0_42 Depth 3
                                        ;       Child Loop BB0_36 Depth 3
                                        ;       Child Loop BB0_53 Depth 3
                                        ;       Child Loop BB0_57 Depth 3
                                        ;       Child Loop BB0_51 Depth 3
	mov.16b	v0, v11
	cmp	w21, #32
	b.ge	LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_6 Depth=2
	mov	w8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	b	LBB0_11
LBB0_8:                                 ;   in Loop: Header=BB0_6 Depth=2
	movi.2d	v1, #0000000000000000
	mov	x8, x22
	ldr	x9, [sp, #128]                  ; 8-byte Folded Reload
	mov	w10, #32                        ; =0x20
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
LBB0_9:                                 ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldp	q5, q6, [x9, #-32]
	ldp	q7, q16, [x8, #-32]
	fmla.8h	v1, v7, v5
	fmla.8h	v2, v16, v6
	ldp	q5, q6, [x9], #64
	ldp	q7, q16, [x8], #64
	fmla.8h	v3, v7, v5
	fmla.8h	v4, v16, v6
	add	x10, x10, #32
	cmp	x10, x26
	b.ls	LBB0_9
; %bb.10:                               ;   in Loop: Header=BB0_6 Depth=2
	fadd.8h	v1, v1, v2
	fadd.8h	v2, v3, v4
	fadd.8h	v1, v1, v2
	ldr	w8, [sp, #140]                  ; 4-byte Folded Reload
LBB0_11:                                ;   in Loop: Header=BB0_6 Depth=2
	ldp	x9, x10, [sp, #152]             ; 16-byte Folded Reload
	madd	x9, x24, x9, x11
	mul	x9, x9, x10
	lsl	x10, x9, #1
	orr	w9, w8, #0x8
	cmp	w9, w21
	stur	x10, [x29, #-144]               ; 8-byte Folded Spill
	b.le	LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_6 Depth=2
	mov	x10, x8
	b	LBB0_15
LBB0_13:                                ;   in Loop: Header=BB0_6 Depth=2
	ldr	x9, [sp, #112]                  ; 8-byte Folded Reload
	add	x9, x9, x10
LBB0_14:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x23, x10]
	ldr	q3, [x9, x10]
	fmla.8h	v1, v3, v2
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w21, w11
	b.ge	LBB0_14
LBB0_15:                                ;   in Loop: Header=BB0_6 Depth=2
	fcvtl	v2.4s, v1.4h
	ext.16b	v1, v1, v1, #8
	fcvtl	v1.4s, v1.4h
	fadd.4s	v1, v2, v1
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	cmp	w10, w21
	b.ge	LBB0_30
; %bb.16:                               ;   in Loop: Header=BB0_6 Depth=2
	mov	w9, w10
	mvn	w8, w10
	add	w10, w21, w8
	cmp	w10, #3
	b.hs	LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_6 Depth=2
	mov	x11, x9
	b	LBB0_28
LBB0_18:                                ;   in Loop: Header=BB0_6 Depth=2
	add	x8, x10, #1
	cmp	w10, #31
	b.hs	LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_6 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB0_24
LBB0_20:                                ;   in Loop: Header=BB0_6 Depth=2
	and	x10, x8, #0x1ffffffe0
	lsl	x12, x9, #1
	ldr	x11, [sp, #72]                  ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x27, x12
	mov	x13, x10
LBB0_21:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	add	x14, x11, x15
	ldp	q3, q4, [x14]
	ldp	q5, q2, [x14, #32]
	fcvtl	v6.4s, v3.4h
	fcvtl2	v3.4s, v3.8h
	fcvtl	v7.4s, v4.4h
	fcvtl2	v4.4s, v4.8h
	fcvtl	v16.4s, v5.4h
	fcvtl2	v5.4s, v5.8h
	ldp	q17, q18, [x12]
	fcvtl	v19.4s, v17.4h
	fcvtl2	v17.4s, v17.8h
	fmul.4s	v17, v3, v17
	fcvtl2	v3.4s, v18.8h
	fmul.4s	v6, v6, v19
	fmul.4s	v4, v4, v3
	ldp	q3, q19, [x12, #32]
	fcvtl	v18.4s, v18.4h
	fmul.4s	v7, v7, v18
	fcvtl2	v18.4s, v3.8h
	fmul.4s	v5, v5, v18
	fcvtl2	v18.4s, v2.8h
	fcvtl	v3.4s, v3.4h
	fmul.4s	v16, v16, v3
	fcvtl2	v3.4s, v19.8h
	fmul.4s	v3, v18, v3
	fcvtl	v2.4s, v2.4h
	fcvtl	v18.4s, v19.4h
	fmul.4s	v2, v2, v18
	mov	s18, v6[1]
	fadd	s1, s1, s6
	fadd	s1, s1, s18
	mov	s18, v6[2]
	fadd	s1, s1, s18
	mov	s6, v6[3]
	fadd	s1, s1, s6
	mov	s6, v17[1]
	fadd	s1, s1, s17
	fadd	s1, s1, s6
	mov	s6, v17[2]
	fadd	s1, s1, s6
	mov	s6, v17[3]
	fadd	s1, s1, s6
	mov	s6, v7[1]
	fadd	s1, s1, s7
	fadd	s1, s1, s6
	mov	s6, v7[2]
	fadd	s1, s1, s6
	mov	s6, v7[3]
	fadd	s1, s1, s6
	mov	s6, v4[1]
	fadd	s1, s1, s4
	fadd	s1, s1, s6
	mov	s6, v4[2]
	fadd	s1, s1, s6
	mov	s4, v4[3]
	fadd	s1, s1, s4
	mov	s4, v16[1]
	fadd	s1, s1, s16
	fadd	s1, s1, s4
	mov	s4, v16[2]
	fadd	s1, s1, s4
	mov	s4, v16[3]
	fadd	s1, s1, s4
	mov	s4, v5[1]
	fadd	s1, s1, s5
	fadd	s1, s1, s4
	mov	s4, v5[2]
	fadd	s1, s1, s4
	mov	s4, v5[3]
	fadd	s1, s1, s4
	mov	s4, v2[1]
	fadd	s1, s1, s2
	fadd	s1, s1, s4
	mov	s4, v2[2]
	fadd	s1, s1, s4
	mov	s2, v2[3]
	fadd	s1, s1, s2
	mov	s2, v3[1]
	fadd	s1, s1, s3
	fadd	s1, s1, s2
	mov	s2, v3[2]
	fadd	s1, s1, s2
	mov	s2, v3[3]
	fadd	s1, s1, s2
	add	x11, x11, #64
	add	x12, x12, #64
	subs	x13, x13, #32
	b.ne	LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_6 Depth=2
	cmp	x8, x10
	b.eq	LBB0_30
; %bb.23:                               ;   in Loop: Header=BB0_6 Depth=2
	tst	x8, #0x1c
	b.eq	LBB0_27
LBB0_24:                                ;   in Loop: Header=BB0_6 Depth=2
	and	x12, x8, #0x1fffffffc
	add	x11, x12, x9
	sub	x13, x10, x12
	add	x9, x10, x9
	lsl	x9, x9, #1
LBB0_25:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	d2, [x20, x9]
	ldr	d3, [x27, x9]
	fcvtl	v2.4s, v2.4h
	fcvtl	v3.4s, v3.4h
	fmul.4s	v2, v2, v3
	mov	s3, v2[3]
	mov	s4, v2[2]
	mov	s5, v2[1]
	fadd	s1, s1, s2
	fadd	s1, s1, s5
	fadd	s1, s1, s4
	fadd	s1, s1, s3
	add	x9, x9, #8
	adds	x13, x13, #4
	b.ne	LBB0_25
; %bb.26:                               ;   in Loop: Header=BB0_6 Depth=2
	cmp	x8, x12
	b.ne	LBB0_28
	b	LBB0_30
LBB0_27:                                ;   in Loop: Header=BB0_6 Depth=2
	add	x11, x10, x9
LBB0_28:                                ;   in Loop: Header=BB0_6 Depth=2
	lsl	x8, x11, #1
LBB0_29:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	h2, [x20, x8]
	ldr	h3, [x27, x8]
	fcvt	s2, h2
	fcvt	s3, h3
	fmadd	s1, s2, s3, s1
	add	w11, w11, #1
	add	x8, x8, #2
	cmp	w21, w11
	b.gt	LBB0_29
LBB0_30:                                ;   in Loop: Header=BB0_6 Depth=2
	fmul	s1, s8, s1
	fcmp	s1, s0
	fcsel	s11, s1, s0, gt
	b.le	LBB0_33
; %bb.31:                               ;   in Loop: Header=BB0_6 Depth=2
	fsub	s0, s0, s1
	bl	_expf
	fmul	s10, s10, s0
	cmp	w21, #8
	b.ge	LBB0_38
; %bb.32:                               ;   in Loop: Header=BB0_6 Depth=2
	mov	w8, #0                          ; =0x0
	fmov	s0, #1.00000000
	fadd	s10, s10, s0
	movi.8h	v1, #60, lsl #8
	ldur	x15, [x29, #-160]               ; 8-byte Folded Reload
	ldur	x11, [x29, #-144]               ; 8-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w21
	b.le	LBB0_35
	b	LBB0_44
LBB0_33:                                ;   in Loop: Header=BB0_6 Depth=2
	fsub	s0, s1, s0
	bl	_expf
                                        ; kill: def $s0 killed $s0 def $q0
	ldur	x15, [x29, #-160]               ; 8-byte Folded Reload
	ldur	x11, [x29, #-144]               ; 8-byte Folded Reload
	fadd	s10, s0, s10
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w21, #32
	b.ge	LBB0_41
LBB0_34:                                ;   in Loop: Header=BB0_6 Depth=2
	mov	w8, #0                          ; =0x0
	orr	w9, w8, #0x8
	cmp	w9, w21
	b.gt	LBB0_44
LBB0_35:                                ;   in Loop: Header=BB0_6 Depth=2
	ldr	x9, [sp, #120]                  ; 8-byte Folded Reload
	add	x9, x9, x11
LBB0_36:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x9, x10]
	ldr	q3, [x19, x10]
	fmla.8h	v3, v1, v2
	str	q3, [x19, x10]
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w21, w11
	b.ge	LBB0_36
; %bb.37:                               ;   in Loop: Header=BB0_6 Depth=2
	cmp	w10, w21
	b.ge	LBB0_5
	b	LBB0_45
LBB0_38:                                ;   in Loop: Header=BB0_6 Depth=2
	fcvt	h0, s0
	mov	x8, x19
	mov	w9, #8                          ; =0x8
	ldur	x11, [x29, #-144]               ; 8-byte Folded Reload
LBB0_39:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q1, [x8]
	fmul.8h	v1, v1, v0[0]
	str	q1, [x8], #16
	add	x9, x9, #8
	cmp	x9, x26
	b.ls	LBB0_39
; %bb.40:                               ;   in Loop: Header=BB0_6 Depth=2
	fmov	s0, #1.00000000
	ldur	x15, [x29, #-160]               ; 8-byte Folded Reload
	fadd	s10, s0, s10
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w21, #32
	b.lt	LBB0_34
LBB0_41:                                ;   in Loop: Header=BB0_6 Depth=2
	ldr	x8, [sp, #104]                  ; 8-byte Folded Reload
	mov	x9, x25
	mov	w10, #32                        ; =0x20
LBB0_42:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldur	q3, [x9, #-32]
	ldp	q4, q5, [x8, #-32]
	fmla.8h	v4, v3, v2[0]
	stur	q4, [x8, #-32]
	ldur	q3, [x9, #-16]
	fmla.8h	v5, v3, v2[0]
	stur	q5, [x8, #-16]
	ldr	q3, [x9]
	ldp	q4, q5, [x8]
	fmla.8h	v4, v3, v2[0]
	str	q4, [x8]
	ldr	q3, [x9, #16]
	fmla.8h	v5, v3, v2[0]
	str	q5, [x8, #16]
	add	x10, x10, #32
	add	x9, x9, #64
	add	x8, x8, #64
	cmp	x10, x26
	b.ls	LBB0_42
; %bb.43:                               ;   in Loop: Header=BB0_6 Depth=2
	ldr	w8, [sp, #140]                  ; 4-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w21
	b.le	LBB0_35
LBB0_44:                                ;   in Loop: Header=BB0_6 Depth=2
	mov	x10, x8
	cmp	w10, w21
	b.ge	LBB0_5
LBB0_45:                                ;   in Loop: Header=BB0_6 Depth=2
	mov	w8, w10
	mvn	w9, w10
	add	w10, w21, w9
	cmp	w10, #3
	b.lo	LBB0_49
; %bb.46:                               ;   in Loop: Header=BB0_6 Depth=2
	ldur	x9, [x29, #-152]                ; 8-byte Folded Reload
	mul	x9, x9, x24
	lsl	x12, x8, #1
	add	x11, x8, x10
	lsl	x11, x11, #1
	ldp	x13, x16, [sp, #80]             ; 16-byte Folded Reload
	add	x13, x13, x11
	ldr	x14, [sp, #96]                  ; 8-byte Folded Reload
	add	x14, x14, x9
	add	x14, x14, x12
	add	x9, x16, x9
	add	x9, x9, x11
	add	x11, x19, x12
	cmp	x11, x9
	ccmp	x14, x13, #2, lo
	b.lo	LBB0_49
; %bb.47:                               ;   in Loop: Header=BB0_6 Depth=2
	add	x9, x10, #1
	cmp	w10, #31
	b.hs	LBB0_52
; %bb.48:                               ;   in Loop: Header=BB0_6 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB0_56
LBB0_49:                                ;   in Loop: Header=BB0_6 Depth=2
	mov	x11, x8
LBB0_50:                                ;   in Loop: Header=BB0_6 Depth=2
	lsl	x8, x11, #1
LBB0_51:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	h1, [x19, x8]
	fcvt	s1, h1
	ldr	h2, [x28, x8]
	fcvt	s2, h2
	fmadd	s1, s2, s0, s1
	fcvt	h1, s1
	str	h1, [x19, x8]
	add	w11, w11, #1
	add	x8, x8, #2
	cmp	w21, w11
	b.gt	LBB0_51
	b	LBB0_5
LBB0_52:                                ;   in Loop: Header=BB0_6 Depth=2
	and	x10, x9, #0x1ffffffe0
	ldr	x11, [sp, #104]                 ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x28, x12
	mov	x13, x10
LBB0_53:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldp	q1, q2, [x11, #-32]
	ldp	q3, q4, [x11]
	fcvtl	v5.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fcvtl	v6.4s, v2.4h
	fcvtl2	v2.4s, v2.8h
	fcvtl	v7.4s, v3.4h
	fcvtl2	v3.4s, v3.8h
	fcvtl	v16.4s, v4.4h
	fcvtl2	v4.4s, v4.8h
	ldp	q17, q18, [x12]
	ldp	q19, q20, [x12, #32]
	fcvtl	v21.4s, v17.4h
	fcvtl2	v17.4s, v17.8h
	fcvtl	v22.4s, v18.4h
	fcvtl2	v18.4s, v18.8h
	fcvtl	v23.4s, v19.4h
	fcvtl2	v19.4s, v19.8h
	fcvtl	v24.4s, v20.4h
	fcvtl2	v20.4s, v20.8h
	fmla.4s	v1, v17, v0[0]
	fmla.4s	v5, v21, v0[0]
	fmla.4s	v2, v18, v0[0]
	fmla.4s	v6, v22, v0[0]
	fmla.4s	v3, v19, v0[0]
	fmla.4s	v7, v23, v0[0]
	fmla.4s	v4, v20, v0[0]
	fmla.4s	v16, v24, v0[0]
	fcvtn	v5.4h, v5.4s
	fcvtn2	v5.8h, v1.4s
	fcvtn	v1.4h, v6.4s
	fcvtn2	v1.8h, v2.4s
	fcvtn	v2.4h, v7.4s
	fcvtn2	v2.8h, v3.4s
	fcvtn	v3.4h, v16.4s
	stp	q5, q1, [x11, #-32]
	fcvtn2	v3.8h, v4.4s
	stp	q2, q3, [x11], #64
	add	x12, x12, #64
	subs	x13, x13, #32
	b.ne	LBB0_53
; %bb.54:                               ;   in Loop: Header=BB0_6 Depth=2
	cmp	x9, x10
	b.eq	LBB0_5
; %bb.55:                               ;   in Loop: Header=BB0_6 Depth=2
	tst	x9, #0x1c
	b.eq	LBB0_59
LBB0_56:                                ;   in Loop: Header=BB0_6 Depth=2
	and	x12, x9, #0x1fffffffc
	add	x11, x12, x8
	sub	x13, x10, x12
	add	x8, x10, x8
	lsl	x8, x8, #1
LBB0_57:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	d1, [x19, x8]
	fcvtl	v1.4s, v1.4h
	ldr	d2, [x28, x8]
	fcvtl	v2.4s, v2.4h
	fmla.4s	v1, v2, v0[0]
	fcvtn	v1.4h, v1.4s
	str	d1, [x19, x8]
	add	x8, x8, #8
	adds	x13, x13, #4
	b.ne	LBB0_57
; %bb.58:                               ;   in Loop: Header=BB0_6 Depth=2
	cmp	x9, x12
	b.eq	LBB0_5
	b	LBB0_50
LBB0_59:                                ;   in Loop: Header=BB0_6 Depth=2
	add	x11, x10, x8
	b	LBB0_50
LBB0_60:                                ;   in Loop: Header=BB0_4 Depth=1
	cmp	w21, #1
	ldr	x24, [sp, #160]                 ; 8-byte Folded Reload
	ldr	x12, [sp, #48]                  ; 8-byte Folded Reload
	b.lt	LBB0_3
; %bb.61:                               ;   in Loop: Header=BB0_4 Depth=1
	mov	x8, #0                          ; =0x0
	mov	x9, x12
	mov	x10, x19
	fdiv	s0, s9, s10
LBB0_62:                                ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q1, [x10], #16
	fcvtl	v2.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v2, v2, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q2, q1, [x9, #-16]
	add	x8, x8, #8
	add	x9, x9, #32
	cmp	x8, x24
	b.lt	LBB0_62
	b	LBB0_3
LBB0_63:
	cmp	w21, #0
	b.le	LBB0_68
; %bb.64:
	mov	x20, #0                         ; =0x0
	mov	w21, w4
	add	x22, x3, #16
	lsl	x23, x26, #2
	mvni.4s	v0, #127, msl #16
	fneg.4s	v2, v0
	stur	q2, [x29, #-144]                ; 16-byte Folded Spill
LBB0_65:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_66 Depth 2
	mov	x0, x19
	ldr	x1, [sp, #24]                   ; 8-byte Folded Reload
	bl	_bzero
	ldur	q2, [x29, #-144]                ; 16-byte Folded Reload
	mov	x8, #0                          ; =0x0
	mov	x9, x22
	mov	x10, x19
LBB0_66:                                ;   Parent Loop BB0_65 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q0, [x10], #16
	fcvtl	v1.4s, v0.4h
	fcvtl2	v0.4s, v0.8h
	fmul.4s	v1, v1, v2
	fmul.4s	v0, v0, v2
	stp	q1, q0, [x9, #-16]
	add	x8, x8, #8
	add	x9, x9, #32
	cmp	x8, x24
	b.lt	LBB0_66
; %bb.67:                               ;   in Loop: Header=BB0_65 Depth=1
	add	x20, x20, #1
	add	x22, x22, x23
	cmp	x20, x21
	b.ne	LBB0_65
	b	LBB0_70
LBB0_68:
	mov	x0, x19
	ldr	x1, [sp, #24]                   ; 8-byte Folded Reload
	mov	x19, x4
	bl	_bzero
	mov	x8, x19
LBB0_69:                                ; =>This Inner Loop Header: Depth=1
	subs	w8, w8, #1
	b.ne	LBB0_69
LBB0_70:
	ldp	x29, x30, [sp, #336]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #320]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #304]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #288]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #272]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #256]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #240]              ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #224]            ; 16-byte Folded Reload
	add	sp, sp, #352
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
