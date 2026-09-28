	.build_version macos, 11, 0	sdk_version 27, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_decode_attention_f16_chunked   ; -- Begin function decode_attention_f16_chunked
	.p2align	2
_decode_attention_f16_chunked:          ; @decode_attention_f16_chunked
	.cfi_startproc
; %bb.0:
	mov	w0, #0                          ; =0x0
	ret
	.cfi_endproc
                                        ; -- End function
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
	b.lt	LBB1_70
; %bb.1:
	mov	x19, x7
	mov	x21, x5
	sxtw	x24, w5
	sbfiz	x8, x5, #1, #32
	str	x8, [sp, #24]                   ; 8-byte Folded Spill
	mov	w26, w5
	cmp	w6, #1
	b.lt	LBB1_63
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
	b	LBB1_4
LBB1_3:                                 ;   in Loop: Header=BB1_4 Depth=1
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
	b.eq	LBB1_70
LBB1_4:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_6 Depth 2
                                        ;       Child Loop BB1_9 Depth 3
                                        ;       Child Loop BB1_14 Depth 3
                                        ;       Child Loop BB1_21 Depth 3
                                        ;       Child Loop BB1_25 Depth 3
                                        ;       Child Loop BB1_29 Depth 3
                                        ;       Child Loop BB1_39 Depth 3
                                        ;       Child Loop BB1_42 Depth 3
                                        ;       Child Loop BB1_36 Depth 3
                                        ;       Child Loop BB1_53 Depth 3
                                        ;       Child Loop BB1_57 Depth 3
                                        ;       Child Loop BB1_51 Depth 3
                                        ;     Child Loop BB1_62 Depth 2
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
	b	LBB1_6
LBB1_5:                                 ;   in Loop: Header=BB1_6 Depth=2
	add	x24, x24, #1
	ldur	x8, [x29, #-152]                ; 8-byte Folded Reload
	add	x22, x22, x8
	add	x27, x27, x8
	add	x25, x25, x8
	add	x28, x28, x8
	ldr	x8, [sp, #144]                  ; 8-byte Folded Reload
	cmp	x24, x8
	ldr	x11, [sp, #168]                 ; 8-byte Folded Reload
	b.eq	LBB1_60
LBB1_6:                                 ;   Parent Loop BB1_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_9 Depth 3
                                        ;       Child Loop BB1_14 Depth 3
                                        ;       Child Loop BB1_21 Depth 3
                                        ;       Child Loop BB1_25 Depth 3
                                        ;       Child Loop BB1_29 Depth 3
                                        ;       Child Loop BB1_39 Depth 3
                                        ;       Child Loop BB1_42 Depth 3
                                        ;       Child Loop BB1_36 Depth 3
                                        ;       Child Loop BB1_53 Depth 3
                                        ;       Child Loop BB1_57 Depth 3
                                        ;       Child Loop BB1_51 Depth 3
	mov.16b	v0, v11
	cmp	w21, #32
	b.ge	LBB1_8
; %bb.7:                                ;   in Loop: Header=BB1_6 Depth=2
	mov	w8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	b	LBB1_11
LBB1_8:                                 ;   in Loop: Header=BB1_6 Depth=2
	movi.2d	v1, #0000000000000000
	mov	x8, x22
	ldr	x9, [sp, #128]                  ; 8-byte Folded Reload
	mov	w10, #32                        ; =0x20
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
LBB1_9:                                 ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.ls	LBB1_9
; %bb.10:                               ;   in Loop: Header=BB1_6 Depth=2
	fadd.8h	v1, v1, v2
	fadd.8h	v2, v3, v4
	fadd.8h	v1, v1, v2
	ldr	w8, [sp, #140]                  ; 4-byte Folded Reload
LBB1_11:                                ;   in Loop: Header=BB1_6 Depth=2
	ldp	x9, x10, [sp, #152]             ; 16-byte Folded Reload
	madd	x9, x24, x9, x11
	mul	x9, x9, x10
	lsl	x10, x9, #1
	orr	w9, w8, #0x8
	cmp	w9, w21
	stur	x10, [x29, #-144]               ; 8-byte Folded Spill
	b.le	LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_6 Depth=2
	mov	x10, x8
	b	LBB1_15
LBB1_13:                                ;   in Loop: Header=BB1_6 Depth=2
	ldr	x9, [sp, #112]                  ; 8-byte Folded Reload
	add	x9, x9, x10
LBB1_14:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x23, x10]
	ldr	q3, [x9, x10]
	fmla.8h	v1, v3, v2
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w21, w11
	b.ge	LBB1_14
LBB1_15:                                ;   in Loop: Header=BB1_6 Depth=2
	fcvtl	v2.4s, v1.4h
	ext.16b	v1, v1, v1, #8
	fcvtl	v1.4s, v1.4h
	fadd.4s	v1, v2, v1
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	cmp	w10, w21
	b.ge	LBB1_30
; %bb.16:                               ;   in Loop: Header=BB1_6 Depth=2
	mov	w9, w10
	mvn	w8, w10
	add	w10, w21, w8
	cmp	w10, #3
	b.hs	LBB1_18
; %bb.17:                               ;   in Loop: Header=BB1_6 Depth=2
	mov	x11, x9
	b	LBB1_28
LBB1_18:                                ;   in Loop: Header=BB1_6 Depth=2
	add	x8, x10, #1
	cmp	w10, #31
	b.hs	LBB1_20
; %bb.19:                               ;   in Loop: Header=BB1_6 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB1_24
LBB1_20:                                ;   in Loop: Header=BB1_6 Depth=2
	and	x10, x8, #0x1ffffffe0
	lsl	x12, x9, #1
	ldr	x11, [sp, #72]                  ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x27, x12
	mov	x13, x10
LBB1_21:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.ne	LBB1_21
; %bb.22:                               ;   in Loop: Header=BB1_6 Depth=2
	cmp	x8, x10
	b.eq	LBB1_30
; %bb.23:                               ;   in Loop: Header=BB1_6 Depth=2
	tst	x8, #0x1c
	b.eq	LBB1_27
LBB1_24:                                ;   in Loop: Header=BB1_6 Depth=2
	and	x12, x8, #0x1fffffffc
	add	x11, x12, x9
	sub	x13, x10, x12
	add	x9, x10, x9
	lsl	x9, x9, #1
LBB1_25:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.ne	LBB1_25
; %bb.26:                               ;   in Loop: Header=BB1_6 Depth=2
	cmp	x8, x12
	b.ne	LBB1_28
	b	LBB1_30
LBB1_27:                                ;   in Loop: Header=BB1_6 Depth=2
	add	x11, x10, x9
LBB1_28:                                ;   in Loop: Header=BB1_6 Depth=2
	lsl	x8, x11, #1
LBB1_29:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	h2, [x20, x8]
	ldr	h3, [x27, x8]
	fcvt	s2, h2
	fcvt	s3, h3
	fmadd	s1, s2, s3, s1
	add	w11, w11, #1
	add	x8, x8, #2
	cmp	w21, w11
	b.gt	LBB1_29
LBB1_30:                                ;   in Loop: Header=BB1_6 Depth=2
	fmul	s1, s8, s1
	fcmp	s1, s0
	fcsel	s11, s1, s0, gt
	b.le	LBB1_33
; %bb.31:                               ;   in Loop: Header=BB1_6 Depth=2
	fsub	s0, s0, s1
	bl	_expf
	fmul	s10, s10, s0
	cmp	w21, #8
	b.ge	LBB1_38
; %bb.32:                               ;   in Loop: Header=BB1_6 Depth=2
	mov	w8, #0                          ; =0x0
	fmov	s0, #1.00000000
	fadd	s10, s10, s0
	movi.8h	v1, #60, lsl #8
	ldur	x15, [x29, #-160]               ; 8-byte Folded Reload
	ldur	x11, [x29, #-144]               ; 8-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w21
	b.le	LBB1_35
	b	LBB1_44
LBB1_33:                                ;   in Loop: Header=BB1_6 Depth=2
	fsub	s0, s1, s0
	bl	_expf
                                        ; kill: def $s0 killed $s0 def $q0
	ldur	x15, [x29, #-160]               ; 8-byte Folded Reload
	ldur	x11, [x29, #-144]               ; 8-byte Folded Reload
	fadd	s10, s0, s10
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w21, #32
	b.ge	LBB1_41
LBB1_34:                                ;   in Loop: Header=BB1_6 Depth=2
	mov	w8, #0                          ; =0x0
	orr	w9, w8, #0x8
	cmp	w9, w21
	b.gt	LBB1_44
LBB1_35:                                ;   in Loop: Header=BB1_6 Depth=2
	ldr	x9, [sp, #120]                  ; 8-byte Folded Reload
	add	x9, x9, x11
LBB1_36:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.ge	LBB1_36
; %bb.37:                               ;   in Loop: Header=BB1_6 Depth=2
	cmp	w10, w21
	b.ge	LBB1_5
	b	LBB1_45
LBB1_38:                                ;   in Loop: Header=BB1_6 Depth=2
	fcvt	h0, s0
	mov	x8, x19
	mov	w9, #8                          ; =0x8
	ldur	x11, [x29, #-144]               ; 8-byte Folded Reload
LBB1_39:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q1, [x8]
	fmul.8h	v1, v1, v0[0]
	str	q1, [x8], #16
	add	x9, x9, #8
	cmp	x9, x26
	b.ls	LBB1_39
; %bb.40:                               ;   in Loop: Header=BB1_6 Depth=2
	fmov	s0, #1.00000000
	ldur	x15, [x29, #-160]               ; 8-byte Folded Reload
	fadd	s10, s0, s10
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w21, #32
	b.lt	LBB1_34
LBB1_41:                                ;   in Loop: Header=BB1_6 Depth=2
	ldr	x8, [sp, #104]                  ; 8-byte Folded Reload
	mov	x9, x25
	mov	w10, #32                        ; =0x20
LBB1_42:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.ls	LBB1_42
; %bb.43:                               ;   in Loop: Header=BB1_6 Depth=2
	ldr	w8, [sp, #140]                  ; 4-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w21
	b.le	LBB1_35
LBB1_44:                                ;   in Loop: Header=BB1_6 Depth=2
	mov	x10, x8
	cmp	w10, w21
	b.ge	LBB1_5
LBB1_45:                                ;   in Loop: Header=BB1_6 Depth=2
	mov	w8, w10
	mvn	w9, w10
	add	w10, w21, w9
	cmp	w10, #3
	b.lo	LBB1_49
; %bb.46:                               ;   in Loop: Header=BB1_6 Depth=2
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
	b.lo	LBB1_49
; %bb.47:                               ;   in Loop: Header=BB1_6 Depth=2
	add	x9, x10, #1
	cmp	w10, #31
	b.hs	LBB1_52
; %bb.48:                               ;   in Loop: Header=BB1_6 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB1_56
LBB1_49:                                ;   in Loop: Header=BB1_6 Depth=2
	mov	x11, x8
LBB1_50:                                ;   in Loop: Header=BB1_6 Depth=2
	lsl	x8, x11, #1
LBB1_51:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.gt	LBB1_51
	b	LBB1_5
LBB1_52:                                ;   in Loop: Header=BB1_6 Depth=2
	and	x10, x9, #0x1ffffffe0
	ldr	x11, [sp, #104]                 ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x28, x12
	mov	x13, x10
LBB1_53:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.ne	LBB1_53
; %bb.54:                               ;   in Loop: Header=BB1_6 Depth=2
	cmp	x9, x10
	b.eq	LBB1_5
; %bb.55:                               ;   in Loop: Header=BB1_6 Depth=2
	tst	x9, #0x1c
	b.eq	LBB1_59
LBB1_56:                                ;   in Loop: Header=BB1_6 Depth=2
	and	x12, x9, #0x1fffffffc
	add	x11, x12, x8
	sub	x13, x10, x12
	add	x8, x10, x8
	lsl	x8, x8, #1
LBB1_57:                                ;   Parent Loop BB1_4 Depth=1
                                        ;     Parent Loop BB1_6 Depth=2
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
	b.ne	LBB1_57
; %bb.58:                               ;   in Loop: Header=BB1_6 Depth=2
	cmp	x9, x12
	b.eq	LBB1_5
	b	LBB1_50
LBB1_59:                                ;   in Loop: Header=BB1_6 Depth=2
	add	x11, x10, x8
	b	LBB1_50
LBB1_60:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	w21, #1
	ldr	x24, [sp, #160]                 ; 8-byte Folded Reload
	ldr	x12, [sp, #48]                  ; 8-byte Folded Reload
	b.lt	LBB1_3
; %bb.61:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	x8, #0                          ; =0x0
	mov	x9, x12
	mov	x10, x19
	fdiv	s0, s9, s10
LBB1_62:                                ;   Parent Loop BB1_4 Depth=1
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
	b.lt	LBB1_62
	b	LBB1_3
LBB1_63:
	cmp	w21, #0
	b.le	LBB1_68
; %bb.64:
	mov	x20, #0                         ; =0x0
	mov	w21, w4
	add	x22, x3, #16
	lsl	x23, x26, #2
	mvni.4s	v0, #127, msl #16
	fneg.4s	v2, v0
	stur	q2, [x29, #-144]                ; 16-byte Folded Spill
LBB1_65:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_66 Depth 2
	mov	x0, x19
	ldr	x1, [sp, #24]                   ; 8-byte Folded Reload
	bl	_bzero
	ldur	q2, [x29, #-144]                ; 16-byte Folded Reload
	mov	x8, #0                          ; =0x0
	mov	x9, x22
	mov	x10, x19
LBB1_66:                                ;   Parent Loop BB1_65 Depth=1
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
	b.lt	LBB1_66
; %bb.67:                               ;   in Loop: Header=BB1_65 Depth=1
	add	x20, x20, #1
	add	x22, x22, x23
	cmp	x20, x21
	b.ne	LBB1_65
	b	LBB1_70
LBB1_68:
	mov	x0, x19
	ldr	x1, [sp, #24]                   ; 8-byte Folded Reload
	mov	x19, x4
	bl	_bzero
	mov	x8, x19
LBB1_69:                                ; =>This Inner Loop Header: Depth=1
	subs	w8, w8, #1
	b.ne	LBB1_69
LBB1_70:
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
	.globl	_decode_attention_f32_q_f16_kv  ; -- Begin function decode_attention_f32_q_f16_kv
	.p2align	2
_decode_attention_f32_q_f16_kv:         ; @decode_attention_f32_q_f16_kv
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #496
	stp	d13, d12, [sp, #352]            ; 16-byte Folded Spill
	stp	d11, d10, [sp, #368]            ; 16-byte Folded Spill
	stp	d9, d8, [sp, #384]              ; 16-byte Folded Spill
	stp	x28, x27, [sp, #400]            ; 16-byte Folded Spill
	stp	x26, x25, [sp, #416]            ; 16-byte Folded Spill
	stp	x24, x23, [sp, #432]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #448]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #464]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #480]            ; 16-byte Folded Spill
	add	x29, sp, #480
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
	.cfi_offset b12, -136
	.cfi_offset b13, -144
	mov	x19, x7
	mov.16b	v8, v0
	mov	x20, x5
	str	x3, [sp, #56]                   ; 8-byte Folded Spill
	stp	x1, x2, [x29, #-232]            ; 16-byte Folded Spill
	ldp	x27, x9, [x29, #16]
	mov	w8, #16959                      ; =0x423f
	movk	w8, #15, lsl #16
	cmp	w6, w8
	str	x9, [sp, #208]                  ; 8-byte Folded Spill
	ccmp	x9, #0, #4, ge
	b.ne	LBB2_68
; %bb.1:
	cmp	w4, #1
	b.lt	LBB2_183
; %bb.2:
	sxtw	x24, w20
	sbfiz	x8, x20, #1, #32
	str	x8, [sp, #168]                  ; 8-byte Folded Spill
	mov	w22, w20
	cmp	w6, #1
	b.lt	LBB2_173
; %bb.3:
	stur	xzr, [x29, #-168]               ; 8-byte Folded Spill
	and	w8, w20, #0x7fffffe0
	stur	w8, [x29, #-200]                ; 4-byte Folded Spill
	mov	w8, w4
	mov	w9, w6
	stur	x9, [x29, #-192]                ; 8-byte Folded Spill
	add	x9, x19, #2
	str	x9, [sp, #224]                  ; 8-byte Folded Spill
	stur	x8, [x29, #-184]                ; 8-byte Folded Spill
	mul	x8, x24, x8
	lsl	x8, x8, #1
	stur	x8, [x29, #-160]                ; 8-byte Folded Spill
	add	x11, x0, #16
	lsl	x8, x24, #2
	str	x8, [sp, #160]                  ; 8-byte Folded Spill
	add	x8, x27, #32
	stur	x8, [x29, #-208]                ; 8-byte Folded Spill
	ldur	x23, [x29, #-232]               ; 8-byte Folded Reload
	add	x8, x23, #32
	str	x8, [sp, #216]                  ; 8-byte Folded Spill
	lsl	x8, x24, #1
	str	x8, [sp, #152]                  ; 8-byte Folded Spill
	ldur	x28, [x29, #-224]               ; 8-byte Folded Reload
	add	x25, x28, #32
	add	x10, x19, #32
	stur	x10, [x29, #-216]               ; 8-byte Folded Spill
	ldr	x10, [sp, #56]                  ; 8-byte Folded Reload
	add	x13, x10, #16
	fmov	s9, #1.00000000
	stur	x24, [x29, #-176]               ; 8-byte Folded Spill
	b	LBB2_5
LBB2_4:                                 ;   in Loop: Header=BB2_5 Depth=1
	ldur	x8, [x29, #-168]                ; 8-byte Folded Reload
	add	x8, x8, #1
	ldp	x12, x9, [sp, #152]             ; 16-byte Folded Reload
	add	x11, x11, x9
	ldr	x10, [sp, #216]                 ; 8-byte Folded Reload
	add	x10, x10, x12
	str	x10, [sp, #216]                 ; 8-byte Folded Spill
	ldp	x23, x28, [sp, #176]            ; 16-byte Folded Reload
	add	x23, x23, x12
	ldr	x25, [sp, #200]                 ; 8-byte Folded Reload
	add	x25, x25, x12
	add	x28, x28, x12
	add	x13, x13, x9
	stur	x8, [x29, #-168]                ; 8-byte Folded Spill
	ldur	x9, [x29, #-184]                ; 8-byte Folded Reload
	cmp	x8, x9
	b.eq	LBB2_183
LBB2_5:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_7 Depth 2
                                        ;     Child Loop BB2_10 Depth 2
                                        ;       Child Loop BB2_13 Depth 3
                                        ;       Child Loop BB2_18 Depth 3
                                        ;       Child Loop BB2_25 Depth 3
                                        ;       Child Loop BB2_29 Depth 3
                                        ;       Child Loop BB2_33 Depth 3
                                        ;       Child Loop BB2_43 Depth 3
                                        ;       Child Loop BB2_46 Depth 3
                                        ;       Child Loop BB2_40 Depth 3
                                        ;       Child Loop BB2_57 Depth 3
                                        ;       Child Loop BB2_61 Depth 3
                                        ;       Child Loop BB2_55 Depth 3
                                        ;     Child Loop BB2_67 Depth 2
	str	x13, [sp, #192]                 ; 8-byte Folded Spill
	cmp	w20, #1
	b.lt	LBB2_8
; %bb.6:                                ;   in Loop: Header=BB2_5 Depth=1
	mov	x8, #0                          ; =0x0
	mov	x9, x27
	mov	x10, x11
LBB2_7:                                 ;   Parent Loop BB2_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x10, #-16]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	str	q0, [x9], #16
	add	x8, x8, #8
	add	x10, x10, #32
	cmp	x8, x24
	b.lt	LBB2_7
LBB2_8:                                 ;   in Loop: Header=BB2_5 Depth=1
	str	x11, [sp, #208]                 ; 8-byte Folded Spill
	ldr	x1, [sp, #168]                  ; 8-byte Folded Reload
	ldur	x8, [x29, #-168]                ; 8-byte Folded Reload
	mul	x21, x1, x8
	mov	x0, x19
	bl	_bzero
	mov	x24, #0                         ; =0x0
	ldur	x8, [x29, #-224]                ; 8-byte Folded Reload
	add	x8, x8, x21
	str	x8, [sp, #240]                  ; 8-byte Folded Spill
	add	x8, x8, #2
	str	x8, [sp, #232]                  ; 8-byte Folded Spill
	movi.2d	v10, #0000000000000000
	mov	w8, #-8388608                   ; =0xff800000
	fmov	s11, w8
	stp	x23, x28, [sp, #176]            ; 16-byte Folded Spill
	str	x25, [sp, #200]                 ; 8-byte Folded Spill
	ldr	x21, [sp, #216]                 ; 8-byte Folded Reload
	b	LBB2_10
LBB2_9:                                 ;   in Loop: Header=BB2_10 Depth=2
	add	x24, x24, #1
	ldur	x8, [x29, #-160]                ; 8-byte Folded Reload
	add	x21, x21, x8
	add	x23, x23, x8
	add	x25, x25, x8
	add	x28, x28, x8
	ldur	x8, [x29, #-192]                ; 8-byte Folded Reload
	cmp	x24, x8
	b.eq	LBB2_64
LBB2_10:                                ;   Parent Loop BB2_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_13 Depth 3
                                        ;       Child Loop BB2_18 Depth 3
                                        ;       Child Loop BB2_25 Depth 3
                                        ;       Child Loop BB2_29 Depth 3
                                        ;       Child Loop BB2_33 Depth 3
                                        ;       Child Loop BB2_43 Depth 3
                                        ;       Child Loop BB2_46 Depth 3
                                        ;       Child Loop BB2_40 Depth 3
                                        ;       Child Loop BB2_57 Depth 3
                                        ;       Child Loop BB2_61 Depth 3
                                        ;       Child Loop BB2_55 Depth 3
	mov.16b	v0, v11
	cmp	w20, #32
	b.ge	LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_10 Depth=2
	mov	w8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	b	LBB2_15
LBB2_12:                                ;   in Loop: Header=BB2_10 Depth=2
	movi.2d	v1, #0000000000000000
	mov	x8, x21
	ldur	x9, [x29, #-208]                ; 8-byte Folded Reload
	mov	w10, #32                        ; =0x20
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
LBB2_13:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
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
	cmp	x10, x22
	b.ls	LBB2_13
; %bb.14:                               ;   in Loop: Header=BB2_10 Depth=2
	fadd.8h	v1, v1, v2
	fadd.8h	v2, v3, v4
	fadd.8h	v1, v1, v2
	ldur	w8, [x29, #-200]                ; 4-byte Folded Reload
LBB2_15:                                ;   in Loop: Header=BB2_10 Depth=2
	ldur	x9, [x29, #-168]                ; 8-byte Folded Reload
	ldp	x11, x10, [x29, #-184]          ; 16-byte Folded Reload
	madd	x9, x24, x11, x9
	mul	x9, x9, x10
	lsl	x10, x9, #1
	orr	w9, w8, #0x8
	cmp	w9, w20
	b.le	LBB2_17
; %bb.16:                               ;   in Loop: Header=BB2_10 Depth=2
	mov	x26, x10
	mov	x10, x8
	b	LBB2_19
LBB2_17:                                ;   in Loop: Header=BB2_10 Depth=2
	ldur	x9, [x29, #-232]                ; 8-byte Folded Reload
	mov	x26, x10
	add	x9, x9, x10
LBB2_18:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x27, x10]
	ldr	q3, [x9, x10]
	fmla.8h	v1, v3, v2
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w20, w11
	b.ge	LBB2_18
LBB2_19:                                ;   in Loop: Header=BB2_10 Depth=2
	fcvtl	v2.4s, v1.4h
	ext.16b	v1, v1, v1, #8
	fcvtl	v1.4s, v1.4h
	fadd.4s	v1, v2, v1
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	cmp	w10, w20
	b.ge	LBB2_34
; %bb.20:                               ;   in Loop: Header=BB2_10 Depth=2
	mov	w9, w10
	mvn	w8, w10
	add	w10, w20, w8
	cmp	w10, #3
	b.hs	LBB2_22
; %bb.21:                               ;   in Loop: Header=BB2_10 Depth=2
	mov	x11, x9
	b	LBB2_32
LBB2_22:                                ;   in Loop: Header=BB2_10 Depth=2
	add	x8, x10, #1
	cmp	w10, #31
	b.hs	LBB2_24
; %bb.23:                               ;   in Loop: Header=BB2_10 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB2_28
LBB2_24:                                ;   in Loop: Header=BB2_10 Depth=2
	and	x10, x8, #0x1ffffffe0
	lsl	x12, x9, #1
	ldur	x11, [x29, #-208]               ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x23, x12
	mov	x13, x10
LBB2_25:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldp	q3, q4, [x11, #-32]
	ldp	q5, q2, [x11], #64
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
	add	x12, x12, #64
	subs	x13, x13, #32
	b.ne	LBB2_25
; %bb.26:                               ;   in Loop: Header=BB2_10 Depth=2
	cmp	x8, x10
	b.eq	LBB2_34
; %bb.27:                               ;   in Loop: Header=BB2_10 Depth=2
	tst	x8, #0x1c
	b.eq	LBB2_31
LBB2_28:                                ;   in Loop: Header=BB2_10 Depth=2
	and	x12, x8, #0x1fffffffc
	add	x11, x12, x9
	sub	x13, x10, x12
	add	x9, x10, x9
	lsl	x9, x9, #1
LBB2_29:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	d2, [x27, x9]
	ldr	d3, [x23, x9]
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
	b.ne	LBB2_29
; %bb.30:                               ;   in Loop: Header=BB2_10 Depth=2
	cmp	x8, x12
	b.ne	LBB2_32
	b	LBB2_34
LBB2_31:                                ;   in Loop: Header=BB2_10 Depth=2
	add	x11, x10, x9
LBB2_32:                                ;   in Loop: Header=BB2_10 Depth=2
	lsl	x8, x11, #1
LBB2_33:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	h2, [x27, x8]
	ldr	h3, [x23, x8]
	fcvt	s2, h2
	fcvt	s3, h3
	fmadd	s1, s2, s3, s1
	add	w11, w11, #1
	add	x8, x8, #2
	cmp	w20, w11
	b.gt	LBB2_33
LBB2_34:                                ;   in Loop: Header=BB2_10 Depth=2
	fmul	s1, s8, s1
	fcmp	s1, s0
	fcsel	s11, s1, s0, gt
	b.le	LBB2_37
; %bb.35:                               ;   in Loop: Header=BB2_10 Depth=2
	fsub	s0, s0, s1
	bl	_expf
	fmul	s10, s10, s0
	cmp	w20, #8
	b.ge	LBB2_42
; %bb.36:                               ;   in Loop: Header=BB2_10 Depth=2
	mov	w8, #0                          ; =0x0
	fmov	s0, #1.00000000
	fadd	s10, s10, s0
	movi.8h	v1, #60, lsl #8
	orr	w9, w8, #0x8
	cmp	w9, w20
	b.le	LBB2_39
	b	LBB2_48
LBB2_37:                                ;   in Loop: Header=BB2_10 Depth=2
	fsub	s0, s1, s0
	bl	_expf
                                        ; kill: def $s0 killed $s0 def $q0
	fadd	s10, s10, s0
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w20, #32
	b.ge	LBB2_45
LBB2_38:                                ;   in Loop: Header=BB2_10 Depth=2
	mov	w8, #0                          ; =0x0
	orr	w9, w8, #0x8
	cmp	w9, w20
	b.gt	LBB2_48
LBB2_39:                                ;   in Loop: Header=BB2_10 Depth=2
	ldur	x9, [x29, #-224]                ; 8-byte Folded Reload
	add	x9, x9, x26
LBB2_40:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x9, x10]
	ldr	q3, [x19, x10]
	fmla.8h	v3, v1, v2
	str	q3, [x19, x10]
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w20, w11
	b.ge	LBB2_40
; %bb.41:                               ;   in Loop: Header=BB2_10 Depth=2
	cmp	w10, w20
	b.ge	LBB2_9
	b	LBB2_49
LBB2_42:                                ;   in Loop: Header=BB2_10 Depth=2
	fcvt	h0, s0
	mov	x8, x19
	mov	w9, #8                          ; =0x8
LBB2_43:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q1, [x8]
	fmul.8h	v1, v1, v0[0]
	str	q1, [x8], #16
	add	x9, x9, #8
	cmp	x9, x22
	b.ls	LBB2_43
; %bb.44:                               ;   in Loop: Header=BB2_10 Depth=2
	fmov	s0, #1.00000000
	fadd	s10, s10, s0
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w20, #32
	b.lt	LBB2_38
LBB2_45:                                ;   in Loop: Header=BB2_10 Depth=2
	ldur	x8, [x29, #-216]                ; 8-byte Folded Reload
	mov	x9, x25
	mov	w10, #32                        ; =0x20
LBB2_46:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
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
	cmp	x10, x22
	b.ls	LBB2_46
; %bb.47:                               ;   in Loop: Header=BB2_10 Depth=2
	ldur	w8, [x29, #-200]                ; 4-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w20
	b.le	LBB2_39
LBB2_48:                                ;   in Loop: Header=BB2_10 Depth=2
	mov	x10, x8
	cmp	w10, w20
	b.ge	LBB2_9
LBB2_49:                                ;   in Loop: Header=BB2_10 Depth=2
	mov	w8, w10
	mvn	w9, w10
	add	w10, w20, w9
	cmp	w10, #3
	b.lo	LBB2_53
; %bb.50:                               ;   in Loop: Header=BB2_10 Depth=2
	ldur	x9, [x29, #-160]                ; 8-byte Folded Reload
	mul	x9, x9, x24
	lsl	x12, x8, #1
	add	x11, x8, x10
	lsl	x11, x11, #1
	ldp	x13, x15, [sp, #224]            ; 16-byte Folded Reload
	add	x13, x13, x11
	ldr	x14, [sp, #240]                 ; 8-byte Folded Reload
	add	x14, x14, x9
	add	x14, x14, x12
	add	x9, x15, x9
	add	x9, x9, x11
	add	x11, x19, x12
	cmp	x11, x9
	ccmp	x14, x13, #2, lo
	b.lo	LBB2_53
; %bb.51:                               ;   in Loop: Header=BB2_10 Depth=2
	add	x9, x10, #1
	cmp	w10, #31
	b.hs	LBB2_56
; %bb.52:                               ;   in Loop: Header=BB2_10 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB2_60
LBB2_53:                                ;   in Loop: Header=BB2_10 Depth=2
	mov	x11, x8
LBB2_54:                                ;   in Loop: Header=BB2_10 Depth=2
	lsl	x8, x11, #1
LBB2_55:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
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
	cmp	w20, w11
	b.gt	LBB2_55
	b	LBB2_9
LBB2_56:                                ;   in Loop: Header=BB2_10 Depth=2
	and	x10, x9, #0x1ffffffe0
	ldur	x11, [x29, #-216]               ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x28, x12
	mov	x13, x10
LBB2_57:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
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
	b.ne	LBB2_57
; %bb.58:                               ;   in Loop: Header=BB2_10 Depth=2
	cmp	x9, x10
	b.eq	LBB2_9
; %bb.59:                               ;   in Loop: Header=BB2_10 Depth=2
	tst	x9, #0x1c
	b.eq	LBB2_63
LBB2_60:                                ;   in Loop: Header=BB2_10 Depth=2
	and	x12, x9, #0x1fffffffc
	add	x11, x12, x8
	sub	x13, x10, x12
	add	x8, x10, x8
	lsl	x8, x8, #1
LBB2_61:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_10 Depth=2
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
	b.ne	LBB2_61
; %bb.62:                               ;   in Loop: Header=BB2_10 Depth=2
	cmp	x9, x12
	b.eq	LBB2_9
	b	LBB2_54
LBB2_63:                                ;   in Loop: Header=BB2_10 Depth=2
	add	x11, x10, x8
	b	LBB2_54
LBB2_64:                                ;   in Loop: Header=BB2_5 Depth=1
	fcmp	s10, #0.0
	ldur	x24, [x29, #-176]               ; 8-byte Folded Reload
	ldr	x11, [sp, #208]                 ; 8-byte Folded Reload
	ldr	x13, [sp, #192]                 ; 8-byte Folded Reload
	b.eq	LBB2_4
; %bb.65:                               ;   in Loop: Header=BB2_5 Depth=1
	cmp	w20, #1
	b.lt	LBB2_4
; %bb.66:                               ;   in Loop: Header=BB2_5 Depth=1
	mov	x8, #0                          ; =0x0
	mov	x9, x13
	mov	x10, x19
	fdiv	s0, s9, s10
LBB2_67:                                ;   Parent Loop BB2_5 Depth=1
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
	b.lt	LBB2_67
	b	LBB2_4
LBB2_68:
	str	w4, [sp, #8]                    ; 4-byte Folded Spill
	ldr	w8, [x29, #32]
	mov	w9, #256                        ; =0x100
	cmp	w8, #1
	csel	w11, w9, w8, lt
	str	w6, [sp, #12]                   ; 4-byte Folded Spill
	add	w8, w6, w11
	sub	w8, w8, #1
	udiv	w23, w8, w11
	add	w8, w20, #2
	str	x8, [sp, #24]                   ; 8-byte Folded Spill
	mov	w28, w20
	cmp	w23, #1
	b.lt	LBB2_147
; %bb.69:
	ldr	w10, [sp, #8]                   ; 4-byte Folded Reload
	cmp	w10, #1
	b.lt	LBB2_180
; %bb.70:
	mov	x24, #0                         ; =0x0
	str	xzr, [sp, #136]                 ; 8-byte Folded Spill
	and	w8, w20, #0x7fffffe0
	stur	w8, [x29, #-208]                ; 4-byte Folded Spill
	sxtw	x25, w20
	sub	w9, w20, #1
	lsr	w9, w9, #3
	ubfiz	x9, x9, #4, #32
	str	x9, [sp, #144]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #24]                   ; 8-byte Folded Reload
	sxtw	x26, w9
	mov	w10, w10
	ldr	w9, [sp, #12]                   ; 4-byte Folded Reload
	mov	w9, w9
	str	x9, [sp, #32]                   ; 8-byte Folded Spill
	add	x9, x19, #2
	str	x9, [sp, #216]                  ; 8-byte Folded Spill
	stur	x10, [x29, #-184]               ; 8-byte Folded Spill
	mul	x9, x25, x10
	mul	x10, x9, x11
	lsl	x10, x10, #1
	str	x10, [sp, #48]                  ; 8-byte Folded Spill
	sbfiz	x10, x20, #1, #32
	str	x10, [sp, #176]                 ; 8-byte Folded Spill
	lsl	x9, x9, #1
	stur	x9, [x29, #-168]                ; 8-byte Folded Spill
	add	x9, x0, #16
	str	x9, [sp, #16]                   ; 8-byte Folded Spill
	sbfiz	x8, x20, #2, #32
	str	x8, [sp, #152]                  ; 8-byte Folded Spill
	add	x8, x27, #32
	stur	x8, [x29, #-216]                ; 8-byte Folded Spill
	ldp	x14, x8, [x29, #-232]           ; 16-byte Folded Reload
	add	x10, x14, #32
	add	x22, x8, #32
	add	x13, x19, #32
	str	x13, [sp, #240]                 ; 8-byte Folded Spill
	mov	x13, x8
	str	x11, [sp, #40]                  ; 8-byte Folded Spill
	str	x23, [sp, #104]                 ; 8-byte Folded Spill
	stur	x25, [x29, #-192]               ; 8-byte Folded Spill
	str	x26, [sp, #96]                  ; 8-byte Folded Spill
	b	LBB2_72
LBB2_71:                                ;   in Loop: Header=BB2_72 Depth=1
	add	x24, x24, #1
	ldr	x11, [sp, #40]                  ; 8-byte Folded Reload
	ldr	x8, [sp, #136]                  ; 8-byte Folded Reload
	add	x8, x8, x11
	str	x8, [sp, #136]                  ; 8-byte Folded Spill
	ldr	x8, [sp, #48]                   ; 8-byte Folded Reload
	ldp	x22, x10, [sp, #80]             ; 16-byte Folded Reload
	add	x10, x10, x8
	ldp	x14, x13, [sp, #64]             ; 16-byte Folded Reload
	add	x14, x14, x8
	add	x22, x22, x8
	add	x13, x13, x8
	cmp	x24, x23
	b.hs	LBB2_147
LBB2_72:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_146 Depth 2
                                        ;     Child Loop BB2_140 Depth 2
                                        ;       Child Loop BB2_141 Depth 3
                                        ;       Child Loop BB2_143 Depth 3
                                        ;     Child Loop BB2_75 Depth 2
                                        ;       Child Loop BB2_77 Depth 3
                                        ;       Child Loop BB2_81 Depth 3
                                        ;         Child Loop BB2_84 Depth 4
                                        ;         Child Loop BB2_89 Depth 4
                                        ;         Child Loop BB2_96 Depth 4
                                        ;         Child Loop BB2_100 Depth 4
                                        ;         Child Loop BB2_104 Depth 4
                                        ;         Child Loop BB2_114 Depth 4
                                        ;         Child Loop BB2_117 Depth 4
                                        ;         Child Loop BB2_111 Depth 4
                                        ;         Child Loop BB2_128 Depth 4
                                        ;         Child Loop BB2_132 Depth 4
                                        ;         Child Loop BB2_126 Depth 4
                                        ;       Child Loop BB2_137 Depth 3
	mul	x8, x24, x11
	ldr	x9, [sp, #32]                   ; 8-byte Folded Reload
	cmp	x8, x9
	stp	x22, x10, [sp, #80]             ; 16-byte Folded Spill
	stp	x14, x13, [sp, #64]             ; 16-byte Folded Spill
	b.hs	LBB2_138
; %bb.73:                               ;   in Loop: Header=BB2_72 Depth=1
	stur	xzr, [x29, #-176]               ; 8-byte Folded Spill
	add	w8, w11, w8
	ldr	w9, [sp, #12]                   ; 4-byte Folded Reload
	cmp	w8, w9
	csel	w8, w8, w9, lt
	sxtw	x8, w8
	stur	x8, [x29, #-200]                ; 8-byte Folded Spill
	ldur	x8, [x29, #-224]                ; 8-byte Folded Reload
	ldr	x9, [sp, #48]                   ; 8-byte Folded Reload
	madd	x8, x9, x24, x8
	stp	x8, x24, [sp, #120]             ; 16-byte Folded Spill
	add	x8, x8, #2
	str	x8, [sp, #112]                  ; 8-byte Folded Spill
	mov	x21, x13
	stp	x10, x14, [sp, #192]            ; 16-byte Folded Spill
	ldr	x8, [sp, #16]                   ; 8-byte Folded Reload
	str	x8, [sp, #184]                  ; 8-byte Folded Spill
	b	LBB2_75
LBB2_74:                                ;   in Loop: Header=BB2_75 Depth=2
	ldur	x9, [x29, #-176]                ; 8-byte Folded Reload
	add	x9, x9, #1
	ldp	x8, x22, [sp, #152]             ; 16-byte Folded Reload
	ldr	x10, [sp, #184]                 ; 8-byte Folded Reload
	add	x10, x10, x8
	str	x10, [sp, #184]                 ; 8-byte Folded Spill
	ldp	x21, x8, [sp, #168]             ; 16-byte Folded Reload
	ldr	x10, [sp, #192]                 ; 8-byte Folded Reload
	add	x10, x10, x8
	str	x10, [sp, #192]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #200]                 ; 8-byte Folded Reload
	add	x10, x10, x8
	str	x10, [sp, #200]                 ; 8-byte Folded Spill
	add	x22, x22, x8
	add	x21, x21, x8
	ldur	x8, [x29, #-184]                ; 8-byte Folded Reload
	stur	x9, [x29, #-176]                ; 8-byte Folded Spill
	cmp	x9, x8
	b.eq	LBB2_71
LBB2_75:                                ;   Parent Loop BB2_72 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_77 Depth 3
                                        ;       Child Loop BB2_81 Depth 3
                                        ;         Child Loop BB2_84 Depth 4
                                        ;         Child Loop BB2_89 Depth 4
                                        ;         Child Loop BB2_96 Depth 4
                                        ;         Child Loop BB2_100 Depth 4
                                        ;         Child Loop BB2_104 Depth 4
                                        ;         Child Loop BB2_114 Depth 4
                                        ;         Child Loop BB2_117 Depth 4
                                        ;         Child Loop BB2_111 Depth 4
                                        ;         Child Loop BB2_128 Depth 4
                                        ;         Child Loop BB2_132 Depth 4
                                        ;         Child Loop BB2_126 Depth 4
                                        ;       Child Loop BB2_137 Depth 3
	cmp	w20, #1
	b.lt	LBB2_79
; %bb.76:                               ;   in Loop: Header=BB2_75 Depth=2
	mov	x8, #0                          ; =0x0
	mov	x9, x27
	ldr	x10, [sp, #184]                 ; 8-byte Folded Reload
LBB2_77:                                ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldp	q0, q1, [x10, #-16]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	str	q0, [x9], #16
	add	x8, x8, #8
	add	x10, x10, #32
	cmp	x8, x25
	b.lt	LBB2_77
; %bb.78:                               ;   in Loop: Header=BB2_75 Depth=2
	ldr	x8, [sp, #144]                  ; 8-byte Folded Reload
	add	x1, x8, #16
	mov	x0, x19
	bl	_bzero
LBB2_79:                                ;   in Loop: Header=BB2_75 Depth=2
	mov	x23, #0                         ; =0x0
	ldr	x8, [sp, #176]                  ; 8-byte Folded Reload
	ldur	x9, [x29, #-176]                ; 8-byte Folded Reload
	mul	x8, x8, x9
	ldr	x9, [sp, #120]                  ; 8-byte Folded Reload
	add	x9, x9, x8
	str	x9, [sp, #232]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #112]                  ; 8-byte Folded Reload
	add	x8, x9, x8
	str	x8, [sp, #224]                  ; 8-byte Folded Spill
	movi.2d	v9, #0000000000000000
	mov	w8, #-8388608                   ; =0xff800000
	fmov	s10, w8
	stp	x22, x21, [sp, #160]            ; 16-byte Folded Spill
	mov	x24, x21
	ldp	x25, x21, [sp, #192]            ; 16-byte Folded Reload
	ldr	x26, [sp, #136]                 ; 8-byte Folded Reload
	b	LBB2_81
LBB2_80:                                ;   in Loop: Header=BB2_81 Depth=3
	add	x26, x26, #1
	add	x23, x23, #1
	ldur	x8, [x29, #-168]                ; 8-byte Folded Reload
	add	x25, x25, x8
	add	x21, x21, x8
	add	x22, x22, x8
	add	x24, x24, x8
	ldur	x8, [x29, #-200]                ; 8-byte Folded Reload
	cmp	x26, x8
	b.ge	LBB2_135
LBB2_81:                                ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB2_84 Depth 4
                                        ;         Child Loop BB2_89 Depth 4
                                        ;         Child Loop BB2_96 Depth 4
                                        ;         Child Loop BB2_100 Depth 4
                                        ;         Child Loop BB2_104 Depth 4
                                        ;         Child Loop BB2_114 Depth 4
                                        ;         Child Loop BB2_117 Depth 4
                                        ;         Child Loop BB2_111 Depth 4
                                        ;         Child Loop BB2_128 Depth 4
                                        ;         Child Loop BB2_132 Depth 4
                                        ;         Child Loop BB2_126 Depth 4
	mov.16b	v0, v10
	cmp	w20, #32
	b.ge	LBB2_83
; %bb.82:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	w8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	b	LBB2_86
LBB2_83:                                ;   in Loop: Header=BB2_81 Depth=3
	movi.2d	v1, #0000000000000000
	mov	x8, x25
	ldur	x9, [x29, #-216]                ; 8-byte Folded Reload
	mov	w10, #32                        ; =0x20
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
LBB2_84:                                ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldp	q5, q6, [x9, #-32]
	ldp	q7, q16, [x8, #-32]
	fmla.8h	v1, v7, v5
	fmla.8h	v2, v16, v6
	ldp	q5, q6, [x9], #64
	ldp	q7, q16, [x8], #64
	fmla.8h	v3, v7, v5
	fmla.8h	v4, v16, v6
	add	x10, x10, #32
	cmp	x10, x28
	b.ls	LBB2_84
; %bb.85:                               ;   in Loop: Header=BB2_81 Depth=3
	fadd.8h	v1, v1, v2
	fadd.8h	v2, v3, v4
	fadd.8h	v1, v1, v2
	ldur	w8, [x29, #-208]                ; 4-byte Folded Reload
LBB2_86:                                ;   in Loop: Header=BB2_81 Depth=3
	ldp	x9, x10, [x29, #-184]           ; 16-byte Folded Reload
	madd	x9, x26, x9, x10
	ldur	x10, [x29, #-192]               ; 8-byte Folded Reload
	mul	x9, x9, x10
	lsl	x10, x9, #1
	orr	w9, w8, #0x8
	cmp	w9, w20
	stur	x10, [x29, #-160]               ; 8-byte Folded Spill
	b.le	LBB2_88
; %bb.87:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	x10, x8
	b	LBB2_90
LBB2_88:                                ;   in Loop: Header=BB2_81 Depth=3
	ldur	x9, [x29, #-232]                ; 8-byte Folded Reload
	add	x9, x9, x10
LBB2_89:                                ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x27, x10]
	ldr	q3, [x9, x10]
	fmla.8h	v1, v3, v2
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w20, w11
	b.ge	LBB2_89
LBB2_90:                                ;   in Loop: Header=BB2_81 Depth=3
	fcvtl	v2.4s, v1.4h
	ext.16b	v1, v1, v1, #8
	fcvtl	v1.4s, v1.4h
	fadd.4s	v1, v2, v1
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	cmp	w10, w20
	b.ge	LBB2_105
; %bb.91:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	w9, w10
	mvn	w8, w10
	add	w10, w20, w8
	cmp	w10, #3
	b.hs	LBB2_93
; %bb.92:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	x11, x9
	b	LBB2_103
LBB2_93:                                ;   in Loop: Header=BB2_81 Depth=3
	add	x8, x10, #1
	cmp	w10, #31
	b.hs	LBB2_95
; %bb.94:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	x10, #0                         ; =0x0
	b	LBB2_99
LBB2_95:                                ;   in Loop: Header=BB2_81 Depth=3
	and	x10, x8, #0x1ffffffe0
	lsl	x12, x9, #1
	ldur	x11, [x29, #-216]               ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x21, x12
	mov	x13, x10
LBB2_96:                                ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldp	q3, q4, [x11, #-32]
	ldp	q5, q2, [x11], #64
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
	add	x12, x12, #64
	subs	x13, x13, #32
	b.ne	LBB2_96
; %bb.97:                               ;   in Loop: Header=BB2_81 Depth=3
	cmp	x8, x10
	b.eq	LBB2_105
; %bb.98:                               ;   in Loop: Header=BB2_81 Depth=3
	tst	x8, #0x1c
	b.eq	LBB2_102
LBB2_99:                                ;   in Loop: Header=BB2_81 Depth=3
	and	x12, x8, #0x1fffffffc
	add	x11, x12, x9
	sub	x13, x10, x12
	add	x9, x10, x9
	lsl	x9, x9, #1
LBB2_100:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d2, [x27, x9]
	ldr	d3, [x21, x9]
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
	b.ne	LBB2_100
; %bb.101:                              ;   in Loop: Header=BB2_81 Depth=3
	cmp	x8, x12
	b.ne	LBB2_103
	b	LBB2_105
LBB2_102:                               ;   in Loop: Header=BB2_81 Depth=3
	add	x11, x10, x9
LBB2_103:                               ;   in Loop: Header=BB2_81 Depth=3
	lsl	x8, x11, #1
LBB2_104:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	h2, [x27, x8]
	ldr	h3, [x21, x8]
	fcvt	s2, h2
	fcvt	s3, h3
	fmadd	s1, s2, s3, s1
	add	w11, w11, #1
	add	x8, x8, #2
	cmp	w20, w11
	b.gt	LBB2_104
LBB2_105:                               ;   in Loop: Header=BB2_81 Depth=3
	fmul	s1, s8, s1
	fcmp	s1, s0
	fcsel	s10, s1, s0, gt
	b.le	LBB2_108
; %bb.106:                              ;   in Loop: Header=BB2_81 Depth=3
	fsub	s0, s0, s1
	bl	_expf
	fmul	s9, s9, s0
	cmp	w20, #8
	b.ge	LBB2_113
; %bb.107:                              ;   in Loop: Header=BB2_81 Depth=3
	mov	w8, #0                          ; =0x0
	fmov	s0, #1.00000000
	fadd	s9, s9, s0
	movi.8h	v1, #60, lsl #8
	ldur	x11, [x29, #-160]               ; 8-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w20
	b.le	LBB2_110
	b	LBB2_119
LBB2_108:                               ;   in Loop: Header=BB2_81 Depth=3
	fsub	s0, s1, s0
	bl	_expf
                                        ; kill: def $s0 killed $s0 def $q0
	ldur	x11, [x29, #-160]               ; 8-byte Folded Reload
	fadd	s9, s0, s9
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w20, #32
	b.ge	LBB2_116
LBB2_109:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	w8, #0                          ; =0x0
	orr	w9, w8, #0x8
	cmp	w9, w20
	b.gt	LBB2_119
LBB2_110:                               ;   in Loop: Header=BB2_81 Depth=3
	ldur	x9, [x29, #-224]                ; 8-byte Folded Reload
	add	x9, x9, x11
LBB2_111:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x9, x10]
	ldr	q3, [x19, x10]
	fmla.8h	v3, v1, v2
	str	q3, [x19, x10]
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w20, w11
	b.ge	LBB2_111
; %bb.112:                              ;   in Loop: Header=BB2_81 Depth=3
	cmp	w10, w20
	b.ge	LBB2_80
	b	LBB2_120
LBB2_113:                               ;   in Loop: Header=BB2_81 Depth=3
	fcvt	h0, s0
	mov	x8, x19
	mov	w9, #8                          ; =0x8
	ldur	x11, [x29, #-160]               ; 8-byte Folded Reload
LBB2_114:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	q1, [x8]
	fmul.8h	v1, v1, v0[0]
	str	q1, [x8], #16
	add	x9, x9, #8
	cmp	x9, x28
	b.ls	LBB2_114
; %bb.115:                              ;   in Loop: Header=BB2_81 Depth=3
	fmov	s0, #1.00000000
	fadd	s9, s0, s9
	fcvt	h2, s0
	dup.8h	v1, v2[0]
	cmp	w20, #32
	b.lt	LBB2_109
LBB2_116:                               ;   in Loop: Header=BB2_81 Depth=3
	ldr	x8, [sp, #240]                  ; 8-byte Folded Reload
	mov	x9, x22
	mov	w10, #32                        ; =0x20
LBB2_117:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
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
	cmp	x10, x28
	b.ls	LBB2_117
; %bb.118:                              ;   in Loop: Header=BB2_81 Depth=3
	ldur	w8, [x29, #-208]                ; 4-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w20
	b.le	LBB2_110
LBB2_119:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	x10, x8
	cmp	w10, w20
	b.ge	LBB2_80
LBB2_120:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	w8, w10
	mvn	w9, w10
	add	w10, w20, w9
	cmp	w10, #3
	b.lo	LBB2_124
; %bb.121:                              ;   in Loop: Header=BB2_81 Depth=3
	ldur	x9, [x29, #-168]                ; 8-byte Folded Reload
	mul	x9, x9, x23
	lsl	x12, x8, #1
	add	x11, x8, x10
	lsl	x11, x11, #1
	ldp	x13, x15, [sp, #216]            ; 16-byte Folded Reload
	add	x13, x13, x11
	ldr	x14, [sp, #232]                 ; 8-byte Folded Reload
	add	x14, x14, x9
	add	x14, x14, x12
	add	x9, x15, x9
	add	x9, x9, x11
	add	x11, x19, x12
	cmp	x11, x9
	ccmp	x14, x13, #2, lo
	b.lo	LBB2_124
; %bb.122:                              ;   in Loop: Header=BB2_81 Depth=3
	add	x9, x10, #1
	cmp	w10, #31
	b.hs	LBB2_127
; %bb.123:                              ;   in Loop: Header=BB2_81 Depth=3
	mov	x10, #0                         ; =0x0
	b	LBB2_131
LBB2_124:                               ;   in Loop: Header=BB2_81 Depth=3
	mov	x11, x8
LBB2_125:                               ;   in Loop: Header=BB2_81 Depth=3
	lsl	x8, x11, #1
LBB2_126:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	h1, [x19, x8]
	fcvt	s1, h1
	ldr	h2, [x24, x8]
	fcvt	s2, h2
	fmadd	s1, s2, s0, s1
	fcvt	h1, s1
	str	h1, [x19, x8]
	add	w11, w11, #1
	add	x8, x8, #2
	cmp	w20, w11
	b.gt	LBB2_126
	b	LBB2_80
LBB2_127:                               ;   in Loop: Header=BB2_81 Depth=3
	and	x10, x9, #0x1ffffffe0
	ldr	x11, [sp, #240]                 ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x24, x12
	mov	x13, x10
LBB2_128:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
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
	b.ne	LBB2_128
; %bb.129:                              ;   in Loop: Header=BB2_81 Depth=3
	cmp	x9, x10
	b.eq	LBB2_80
; %bb.130:                              ;   in Loop: Header=BB2_81 Depth=3
	tst	x9, #0x1c
	b.eq	LBB2_134
LBB2_131:                               ;   in Loop: Header=BB2_81 Depth=3
	and	x12, x9, #0x1fffffffc
	add	x11, x12, x8
	sub	x13, x10, x12
	add	x8, x10, x8
	lsl	x8, x8, #1
LBB2_132:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ;       Parent Loop BB2_81 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d1, [x19, x8]
	fcvtl	v1.4s, v1.4h
	ldr	d2, [x24, x8]
	fcvtl	v2.4s, v2.4h
	fmla.4s	v1, v2, v0[0]
	fcvtn	v1.4h, v1.4s
	str	d1, [x19, x8]
	add	x8, x8, #8
	adds	x13, x13, #4
	b.ne	LBB2_132
; %bb.133:                              ;   in Loop: Header=BB2_81 Depth=3
	cmp	x9, x12
	b.eq	LBB2_80
	b	LBB2_125
LBB2_134:                               ;   in Loop: Header=BB2_81 Depth=3
	add	x11, x10, x8
	b	LBB2_125
LBB2_135:                               ;   in Loop: Header=BB2_75 Depth=2
	ldp	x26, x23, [sp, #96]             ; 16-byte Folded Reload
	ldr	x24, [sp, #128]                 ; 8-byte Folded Reload
	ldur	x8, [x29, #-176]                ; 8-byte Folded Reload
	madd	x8, x8, x23, x24
	mul	x8, x8, x26
	ldr	x9, [sp, #208]                  ; 8-byte Folded Reload
	add	x9, x9, x8, lsl #2
	stp	s10, s9, [x9], #8
	cmp	w20, #1
	ldur	x25, [x29, #-192]               ; 8-byte Folded Reload
	b.lt	LBB2_74
; %bb.136:                              ;   in Loop: Header=BB2_75 Depth=2
	mov	x8, #0                          ; =0x0
	mov	x10, x19
LBB2_137:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_75 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q0, [x10], #16
	fcvtl	v1.4s, v0.4h
	fcvtl2	v0.4s, v0.8h
	add	x11, x9, x8, lsl #2
	stp	q1, q0, [x11]
	add	x8, x8, #8
	cmp	x8, x25
	b.lt	LBB2_137
	b	LBB2_74
LBB2_138:                               ;   in Loop: Header=BB2_72 Depth=1
	cmp	w20, #0
	b.le	LBB2_145
; %bb.139:                              ;   in Loop: Header=BB2_72 Depth=1
	mov	x21, #0                         ; =0x0
	ldr	x22, [sp, #16]                  ; 8-byte Folded Reload
LBB2_140:                               ;   Parent Loop BB2_72 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_141 Depth 3
                                        ;       Child Loop BB2_143 Depth 3
	mov	x8, #0                          ; =0x0
	mov	x9, x22
	mov	x10, x27
LBB2_141:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_140 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldp	q0, q1, [x9, #-16]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	str	q0, [x10], #16
	add	x8, x8, #8
	add	x9, x9, #32
	cmp	x8, x25
	b.lt	LBB2_141
; %bb.142:                              ;   in Loop: Header=BB2_140 Depth=2
	ldr	x8, [sp, #144]                  ; 8-byte Folded Reload
	add	x1, x8, #16
	mov	x0, x19
	bl	_bzero
	mov	x8, #0                          ; =0x0
	madd	x9, x21, x23, x24
	mul	x9, x9, x26
	ldr	x10, [sp, #208]                 ; 8-byte Folded Reload
	add	x9, x10, x9, lsl #2
	mov	w10, #-8388608                  ; =0xff800000
	str	x10, [x9], #8
	mov	x10, x19
LBB2_143:                               ;   Parent Loop BB2_72 Depth=1
                                        ;     Parent Loop BB2_140 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q0, [x10], #16
	fcvtl	v1.4s, v0.4h
	fcvtl2	v0.4s, v0.8h
	add	x11, x9, x8, lsl #2
	stp	q1, q0, [x11]
	add	x8, x8, #8
	cmp	x8, x25
	b.lt	LBB2_143
; %bb.144:                              ;   in Loop: Header=BB2_140 Depth=2
	add	x21, x21, #1
	ldr	x8, [sp, #152]                  ; 8-byte Folded Reload
	add	x22, x22, x8
	ldur	x8, [x29, #-184]                ; 8-byte Folded Reload
	cmp	x21, x8
	b.ne	LBB2_140
	b	LBB2_71
LBB2_145:                               ;   in Loop: Header=BB2_72 Depth=1
	mov	x8, #0                          ; =0x0
	ldur	x9, [x29, #-184]                ; 8-byte Folded Reload
LBB2_146:                               ;   Parent Loop BB2_72 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	madd	x10, x8, x23, x24
	mul	x10, x10, x26
	lsl	x10, x10, #2
	ldr	x11, [sp, #208]                 ; 8-byte Folded Reload
	mov	w12, #-8388608                  ; =0xff800000
	str	x12, [x11, x10]
	add	x8, x8, #1
	subs	x9, x9, #1
	b.ne	LBB2_146
	b	LBB2_71
LBB2_147:
	ldr	w8, [sp, #8]                    ; 4-byte Folded Reload
	cmp	w8, #1
	b.lt	LBB2_180
; %bb.148:
	sbfiz	x1, x20, #2, #32
	mov	w21, w8
	cmp	w23, #1
	b.lt	LBB2_178
; %bb.149:
	sxtw	x24, w20
	ldr	x8, [sp, #24]                   ; 8-byte Folded Reload
	sxtw	x25, w8
	sxtw	x26, w23
	add	x10, x19, w20, uxtw #2
	smull	x8, w23, w8
	mov	x23, #0                         ; =0x0
	lsl	x8, x8, #2
	stp	x8, x10, [x29, #-232]           ; 16-byte Folded Spill
	sub	x8, x26, #1
	madd	x8, x8, x25, x28
	ldr	x9, [sp, #208]                  ; 8-byte Folded Reload
	add	x9, x9, #8
	add	x8, x9, x8, lsl #2
	stp	x8, x9, [sp, #232]              ; 16-byte Folded Spill
	and	x8, x28, #0x7ffffff0
	stur	x8, [x29, #-168]                ; 8-byte Folded Spill
	and	x8, x28, #0xc
	str	x8, [sp, #216]                  ; 8-byte Folded Spill
	and	x8, x28, #0x7ffffffc
	add	x9, x19, #32
	stur	x9, [x29, #-200]                ; 8-byte Folded Spill
	stur	x8, [x29, #-176]                ; 8-byte Folded Spill
	neg	x8, x8
	stur	x8, [x29, #-184]                ; 8-byte Folded Spill
	lsl	x8, x24, #2
	str	x8, [sp, #224]                  ; 8-byte Folded Spill
	fmov	s8, #1.00000000
	stp	x1, x21, [x29, #-216]           ; 16-byte Folded Spill
	b	LBB2_151
LBB2_150:                               ;   in Loop: Header=BB2_151 Depth=1
	ldur	x23, [x29, #-192]               ; 8-byte Folded Reload
	add	x23, x23, #1
	ldr	x8, [sp, #224]                  ; 8-byte Folded Reload
	add	x11, x11, x8
	str	x11, [sp, #56]                  ; 8-byte Folded Spill
	ldp	x1, x21, [x29, #-216]           ; 16-byte Folded Reload
	cmp	x23, x21
	b.eq	LBB2_180
LBB2_151:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_154 Depth 2
                                        ;       Child Loop BB2_161 Depth 3
                                        ;       Child Loop BB2_165 Depth 3
                                        ;       Child Loop BB2_168 Depth 3
                                        ;     Child Loop BB2_172 Depth 2
	mov	x0, x19
	bl	_bzero
	mov	x22, #0                         ; =0x0
	mul	x27, x23, x26
	stur	x23, [x29, #-192]               ; 8-byte Folded Spill
	ldur	x8, [x29, #-232]                ; 8-byte Folded Reload
	mul	x8, x8, x23
	ldp	x10, x9, [sp, #232]             ; 16-byte Folded Reload
	add	x9, x9, x8
	add	x8, x10, x8
	cmp	x19, x8
	ldur	x8, [x29, #-224]                ; 8-byte Folded Reload
	ccmp	x9, x8, #2, lo
	ldr	x8, [sp, #24]                   ; 8-byte Folded Reload
	ccmp	w8, #0, #8, hs
	ccmp	w20, #4, #0, ge
	cset	w21, lo
	movi.2d	v9, #0000000000000000
	mov	w8, #-8388608                   ; =0xff800000
	fmov	s0, w8
	b	LBB2_154
LBB2_152:                               ;   in Loop: Header=BB2_154 Depth=2
	fmul	s0, s10, s0
	fmadd	s9, s9, s17, s0
	mov.16b	v0, v11
LBB2_153:                               ;   in Loop: Header=BB2_154 Depth=2
	add	x22, x22, #1
	cmp	x22, x26
	b.ge	LBB2_169
LBB2_154:                               ;   Parent Loop BB2_151 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_161 Depth 3
                                        ;       Child Loop BB2_165 Depth 3
                                        ;       Child Loop BB2_168 Depth 3
	add	x8, x22, x27
	mul	x8, x8, x25
	ldr	x9, [sp, #208]                  ; 8-byte Folded Reload
	add	x23, x9, x8, lsl #2
	ldr	s10, [x23, #4]
	fcmp	s10, #0.0
	b.eq	LBB2_153
; %bb.155:                              ;   in Loop: Header=BB2_154 Depth=2
	ldr	s12, [x23]
	fmaxnm	s11, s0, s12
	fsub	s0, s0, s11
	bl	_expf
                                        ; kill: def $s0 killed $s0 def $q0
	stur	q0, [x29, #-160]                ; 16-byte Folded Spill
	fsub	s0, s12, s11
	bl	_expf
	ldur	q17, [x29, #-160]               ; 16-byte Folded Reload
                                        ; kill: def $s0 killed $s0 def $q0
	cmp	w20, #1
	b.lt	LBB2_152
; %bb.156:                              ;   in Loop: Header=BB2_154 Depth=2
	add	x8, x23, #8
	tbz	w21, #0, LBB2_158
; %bb.157:                              ;   in Loop: Header=BB2_154 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB2_167
LBB2_158:                               ;   in Loop: Header=BB2_154 Depth=2
	cmp	w20, #16
	b.hs	LBB2_160
; %bb.159:                              ;   in Loop: Header=BB2_154 Depth=2
	mov	x9, #0                          ; =0x0
	b	LBB2_164
LBB2_160:                               ;   in Loop: Header=BB2_154 Depth=2
	mov	x9, #0                          ; =0x0
	ldur	x10, [x29, #-200]               ; 8-byte Folded Reload
	ldur	x11, [x29, #-168]               ; 8-byte Folded Reload
LBB2_161:                               ;   Parent Loop BB2_151 Depth=1
                                        ;     Parent Loop BB2_154 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldp	q1, q2, [x10, #-32]
	ldp	q3, q4, [x10]
	add	x12, x8, x9, lsl #2
	ldp	q5, q6, [x12]
	ldp	q7, q16, [x12, #32]
	fmul.4s	v5, v5, v0[0]
	fmul.4s	v6, v6, v0[0]
	fmul.4s	v7, v7, v0[0]
	fmul.4s	v16, v16, v0[0]
	fmla.4s	v5, v1, v17[0]
	fmla.4s	v6, v2, v17[0]
	fmla.4s	v7, v3, v17[0]
	fmla.4s	v16, v4, v17[0]
	stp	q5, q6, [x10, #-32]
	stp	q7, q16, [x10], #64
	add	x9, x9, #16
	subs	x11, x11, #16
	b.ne	LBB2_161
; %bb.162:                              ;   in Loop: Header=BB2_154 Depth=2
	ldur	x9, [x29, #-168]                ; 8-byte Folded Reload
	cmp	x9, x28
	b.eq	LBB2_152
; %bb.163:                              ;   in Loop: Header=BB2_154 Depth=2
	ldur	x10, [x29, #-168]               ; 8-byte Folded Reload
	mov	x9, x10
	ldr	x11, [sp, #216]                 ; 8-byte Folded Reload
	cbz	x11, LBB2_167
LBB2_164:                               ;   in Loop: Header=BB2_154 Depth=2
	ldur	x10, [x29, #-184]               ; 8-byte Folded Reload
	add	x10, x10, x9
	add	x11, x19, x9, lsl #2
LBB2_165:                               ;   Parent Loop BB2_151 Depth=1
                                        ;     Parent Loop BB2_154 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q1, [x11]
	lsl	x12, x9, #2
	ldr	q2, [x8, x12]
	fmul.4s	v2, v2, v0[0]
	fmla.4s	v2, v1, v17[0]
	str	q2, [x11], #16
	add	x9, x9, #4
	adds	x10, x10, #4
	b.ne	LBB2_165
; %bb.166:                              ;   in Loop: Header=BB2_154 Depth=2
	ldur	x9, [x29, #-176]                ; 8-byte Folded Reload
	mov	x10, x9
	cmp	x9, x28
	b.eq	LBB2_152
LBB2_167:                               ;   in Loop: Header=BB2_154 Depth=2
	sub	x9, x28, x10
	add	x11, x19, x10, lsl #2
LBB2_168:                               ;   Parent Loop BB2_151 Depth=1
                                        ;     Parent Loop BB2_154 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	s1, [x11]
	ldr	s2, [x8, x10, lsl #2]
	fmul	s2, s0, s2
	fmadd	s1, s1, s17, s2
	str	s1, [x11], #4
	add	x10, x10, #1
	subs	x9, x9, #1
	b.ne	LBB2_168
	b	LBB2_152
LBB2_169:                               ;   in Loop: Header=BB2_151 Depth=1
	fcmp	s9, #0.0
	ldr	x11, [sp, #56]                  ; 8-byte Folded Reload
	b.eq	LBB2_150
; %bb.170:                              ;   in Loop: Header=BB2_151 Depth=1
	cmp	w20, #1
	b.lt	LBB2_150
; %bb.171:                              ;   in Loop: Header=BB2_151 Depth=1
	mov	x8, #0                          ; =0x0
	mov	x9, x19
	mov	x10, x11
	fdiv	s0, s8, s9
LBB2_172:                               ;   Parent Loop BB2_151 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q1, [x9], #16
	fmul.4s	v1, v1, v0[0]
	str	q1, [x10], #16
	add	x8, x8, #4
	cmp	x8, x24
	b.lt	LBB2_172
	b	LBB2_150
LBB2_173:
	cmp	w20, #0
	b.le	LBB2_181
; %bb.174:
	mov	x20, #0                         ; =0x0
	add	x21, x0, #16
	lsl	x22, x22, #2
	mov	w23, w4
LBB2_175:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_176 Depth 2
	mov	x8, #0                          ; =0x0
	mov	x9, x27
	mov	x10, x21
LBB2_176:                               ;   Parent Loop BB2_175 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x10, #-16]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	str	q0, [x9], #16
	add	x8, x8, #8
	add	x10, x10, #32
	cmp	x8, x24
	b.lt	LBB2_176
; %bb.177:                              ;   in Loop: Header=BB2_175 Depth=1
	mov	x0, x19
	ldr	x1, [sp, #168]                  ; 8-byte Folded Reload
	bl	_bzero
	add	x20, x20, #1
	add	x21, x21, x22
	cmp	x20, x23
	b.ne	LBB2_175
	b	LBB2_183
LBB2_178:
	mov	x0, x19
	bl	_bzero
LBB2_179:                               ; =>This Inner Loop Header: Depth=1
	subs	x21, x21, #1
	b.ne	LBB2_179
LBB2_180:
	mov	w0, #1                          ; =0x1
	b	LBB2_184
LBB2_181:
	mov	x0, x19
	ldr	x1, [sp, #168]                  ; 8-byte Folded Reload
	mov	x19, x4
	bl	_bzero
	mov	x8, x19
LBB2_182:                               ; =>This Inner Loop Header: Depth=1
	subs	w8, w8, #1
	b.ne	LBB2_182
LBB2_183:
	mov	w0, #0                          ; =0x0
LBB2_184:
	ldp	x29, x30, [sp, #480]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #464]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #448]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #432]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #416]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #400]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #384]              ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #368]            ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #352]            ; 16-byte Folded Reload
	add	sp, sp, #496
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
