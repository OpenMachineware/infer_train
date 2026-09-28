	.build_version macos, 27, 0	sdk_version 27, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_our_attention                  ; -- Begin function our_attention
	.p2align	2
_our_attention:                         ; @our_attention
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #272
	stp	d11, d10, [sp, #144]            ; 16-byte Folded Spill
	stp	d9, d8, [sp, #160]              ; 16-byte Folded Spill
	stp	x28, x27, [sp, #176]            ; 16-byte Folded Spill
	stp	x26, x25, [sp, #192]            ; 16-byte Folded Spill
	stp	x24, x23, [sp, #208]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #224]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #240]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #256]            ; 16-byte Folded Spill
	add	x29, sp, #256
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
	str	x7, [sp, #128]                  ; 8-byte Folded Spill
	str	x3, [sp, #8]                    ; 8-byte Folded Spill
	str	x1, [sp, #88]                   ; 8-byte Folded Spill
	cmp	w4, #1
	b.lt	LBB0_55
; %bb.1:
	mov	x22, x5
	ldr	x25, [x29, #16]
	sxtw	x26, w5
	sbfiz	x8, x5, #1, #32
	str	x8, [sp, #32]                   ; 8-byte Folded Spill
	mov	w27, w5
	cmp	w6, #1
	b.lt	LBB0_46
; %bb.2:
	mov.16b	v8, v0
	mov	x24, x2
	and	w8, w22, #0x7fffffe0
	str	w8, [sp, #84]                   ; 4-byte Folded Spill
	mov	w8, w4
	mov	w23, w6
	add	x11, x0, #16
	lsl	x9, x26, #2
	str	x9, [sp, #24]                   ; 8-byte Folded Spill
	add	x9, x25, #32
	stp	xzr, x9, [sp, #104]             ; 16-byte Folded Spill
	ldr	x28, [sp, #88]                  ; 8-byte Folded Reload
	add	x9, x28, #32
	lsl	x10, x26, #1
	str	x10, [sp, #16]                  ; 8-byte Folded Spill
	str	x8, [sp, #96]                   ; 8-byte Folded Spill
	mul	x8, x26, x8
	lsl	x19, x8, #1
	ldr	x8, [sp, #8]                    ; 8-byte Folded Reload
	add	x8, x8, #16
	stp	x8, x9, [sp, #64]               ; 16-byte Folded Spill
	fmov	s9, #1.00000000
	b	LBB0_4
LBB0_3:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldr	x8, [sp, #104]                  ; 8-byte Folded Reload
	add	x8, x8, #1
	ldp	x28, x11, [sp, #40]             ; 16-byte Folded Reload
	ldp	x12, x9, [sp, #16]              ; 16-byte Folded Reload
	add	x11, x11, x9
	ldr	x10, [sp, #72]                  ; 8-byte Folded Reload
	add	x10, x10, x12
	str	x10, [sp, #72]                  ; 8-byte Folded Spill
	add	x28, x28, x12
	ldp	x24, x10, [sp, #56]             ; 16-byte Folded Reload
	add	x24, x24, x12
	add	x10, x10, x9
	str	x10, [sp, #64]                  ; 8-byte Folded Spill
	str	x8, [sp, #104]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #96]                   ; 8-byte Folded Reload
	cmp	x8, x9
	b.eq	LBB0_55
LBB0_4:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_6 Depth 2
                                        ;     Child Loop BB0_11 Depth 2
                                        ;       Child Loop BB0_16 Depth 3
                                        ;       Child Loop BB0_14 Depth 3
                                        ;       Child Loop BB0_25 Depth 3
                                        ;       Child Loop BB0_29 Depth 3
                                        ;       Child Loop BB0_33 Depth 3
                                        ;       Child Loop BB0_37 Depth 3
                                        ;       Child Loop BB0_41 Depth 3
                                        ;     Child Loop BB0_45 Depth 2
	cmp	w22, #1
	b.lt	LBB0_7
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	mov	x8, #0                          ; =0x0
	mov	x9, x25
	mov	x10, x11
LBB0_6:                                 ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x10, #-16]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	str	q0, [x9], #16
	add	x8, x8, #8
	add	x10, x10, #32
	cmp	x8, x26
	b.lt	LBB0_6
LBB0_7:                                 ;   in Loop: Header=BB0_4 Depth=1
	str	x11, [sp, #48]                  ; 8-byte Folded Spill
	ldr	x0, [sp, #128]                  ; 8-byte Folded Reload
	ldr	x1, [sp, #32]                   ; 8-byte Folded Reload
	bl	_bzero
	mov	x21, #0                         ; =0x0
	mov	w8, #-8388608                   ; =0xff800000
	movi.2d	v10, #0000000000000000
	fmov	s0, w8
	str	x24, [sp, #56]                  ; 8-byte Folded Spill
	str	x28, [sp, #40]                  ; 8-byte Folded Spill
	ldr	x20, [sp, #72]                  ; 8-byte Folded Reload
	b	LBB0_11
LBB0_8:                                 ;   in Loop: Header=BB0_11 Depth=2
	fadd	s10, s1, s9
LBB0_9:                                 ;   in Loop: Header=BB0_11 Depth=2
	add	x21, x21, #1
	cmp	x21, x23
	b.eq	LBB0_43
LBB0_10:                                ;   in Loop: Header=BB0_11 Depth=2
	add	x20, x20, x19
	add	x28, x28, x19
	add	x24, x24, x19
	mov.16b	v0, v11
LBB0_11:                                ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_16 Depth 3
                                        ;       Child Loop BB0_14 Depth 3
                                        ;       Child Loop BB0_25 Depth 3
                                        ;       Child Loop BB0_29 Depth 3
                                        ;       Child Loop BB0_33 Depth 3
                                        ;       Child Loop BB0_37 Depth 3
                                        ;       Child Loop BB0_41 Depth 3
	cmp	w22, #32
	b.ge	LBB0_15
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=2
	mov	w8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	orr	w9, w8, #0x8
	cmp	w9, w22
	b.gt	LBB0_18
LBB0_13:                                ;   in Loop: Header=BB0_11 Depth=2
	ldp	x10, x9, [sp, #96]              ; 16-byte Folded Reload
	madd	x9, x21, x10, x9
	mul	x9, x9, x26
	ldr	x10, [sp, #88]                  ; 8-byte Folded Reload
	add	x9, x10, x9, lsl #1
LBB0_14:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ubfiz	x10, x8, #1, #32
	ldr	q2, [x25, x10]
	ldr	q3, [x9, x10]
	fmla.8h	v1, v3, v2
	add	w10, w8, #8
	add	w11, w8, #16
	mov	x8, x10
	cmp	w22, w11
	b.ge	LBB0_14
	b	LBB0_19
LBB0_15:                                ;   in Loop: Header=BB0_11 Depth=2
	movi.2d	v1, #0000000000000000
	mov	x8, x20
	ldr	x9, [sp, #112]                  ; 8-byte Folded Reload
	mov	w10, #32                        ; =0x20
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
LBB0_16:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
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
	cmp	x10, x27
	b.ls	LBB0_16
; %bb.17:                               ;   in Loop: Header=BB0_11 Depth=2
	fadd.8h	v1, v1, v2
	fadd.8h	v2, v3, v4
	fadd.8h	v1, v1, v2
	ldr	w8, [sp, #84]                   ; 4-byte Folded Reload
	orr	w9, w8, #0x8
	cmp	w9, w22
	b.le	LBB0_13
LBB0_18:                                ;   in Loop: Header=BB0_11 Depth=2
	mov	x10, x8
LBB0_19:                                ;   in Loop: Header=BB0_11 Depth=2
	fcvtl	v2.4s, v1.4h
	ext.16b	v1, v1, v1, #8
	fcvtl	v1.4s, v1.4h
	fadd.4s	v1, v2, v1
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	cmp	w10, w22
	b.ge	LBB0_34
; %bb.20:                               ;   in Loop: Header=BB0_11 Depth=2
	mov	w9, w10
	mvn	w8, w10
	add	w10, w22, w8
	cmp	w10, #3
	b.hs	LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_11 Depth=2
	mov	x11, x9
	b	LBB0_32
LBB0_22:                                ;   in Loop: Header=BB0_11 Depth=2
	add	x8, x10, #1
	cmp	w10, #31
	b.hs	LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_11 Depth=2
	mov	x10, #0                         ; =0x0
	b	LBB0_28
LBB0_24:                                ;   in Loop: Header=BB0_11 Depth=2
	and	x10, x8, #0x1ffffffe0
	lsl	x12, x9, #1
	ldr	x11, [sp, #112]                 ; 8-byte Folded Reload
	add	x11, x11, x12
	add	x12, x28, x12
	mov	x13, x10
LBB0_25:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
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
	b.ne	LBB0_25
; %bb.26:                               ;   in Loop: Header=BB0_11 Depth=2
	cmp	x8, x10
	b.eq	LBB0_34
; %bb.27:                               ;   in Loop: Header=BB0_11 Depth=2
	tst	x8, #0x1c
	b.eq	LBB0_31
LBB0_28:                                ;   in Loop: Header=BB0_11 Depth=2
	and	x12, x8, #0x1fffffffc
	add	x11, x12, x9
	sub	x13, x10, x12
	add	x9, x10, x9
	lsl	x9, x9, #1
LBB0_29:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	d2, [x25, x9]
	ldr	d3, [x28, x9]
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
	b.ne	LBB0_29
; %bb.30:                               ;   in Loop: Header=BB0_11 Depth=2
	cmp	x8, x12
	b.ne	LBB0_32
	b	LBB0_34
LBB0_31:                                ;   in Loop: Header=BB0_11 Depth=2
	add	x11, x10, x9
LBB0_32:                                ;   in Loop: Header=BB0_11 Depth=2
	lsl	x8, x11, #1
LBB0_33:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	h2, [x25, x8]
	ldr	h3, [x28, x8]
	fcvt	s2, h2
	fcvt	s3, h3
	fmadd	s1, s2, s3, s1
	add	w11, w11, #1
	add	x8, x8, #2
	cmp	w22, w11
	b.gt	LBB0_33
LBB0_34:                                ;   in Loop: Header=BB0_11 Depth=2
	fmul	s1, s8, s1
	fcmp	s1, s0
	fcsel	s11, s1, s0, gt
	b.le	LBB0_39
; %bb.35:                               ;   in Loop: Header=BB0_11 Depth=2
	fsub	s0, s0, s1
	bl	_expf
	fmul	s1, s10, s0
	cmp	w22, #0
	b.le	LBB0_8
; %bb.36:                               ;   in Loop: Header=BB0_11 Depth=2
	mov	x8, #0                          ; =0x0
	fcvt	h0, s0
	ldr	x9, [sp, #128]                  ; 8-byte Folded Reload
LBB0_37:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q2, [x9]
	fmul.8h	v2, v2, v0[0]
	str	q2, [x9], #16
	add	x8, x8, #8
	cmp	x8, x27
	b.lo	LBB0_37
; %bb.38:                               ;   in Loop: Header=BB0_11 Depth=2
	fmov	s0, #1.00000000
	fadd	s10, s1, s0
	b	LBB0_40
LBB0_39:                                ;   in Loop: Header=BB0_11 Depth=2
	fsub	s0, s1, s0
	bl	_expf
	fadd	s10, s10, s0
	cmp	w22, #0
	b.le	LBB0_9
LBB0_40:                                ;   in Loop: Header=BB0_11 Depth=2
	mov	x8, #0                          ; =0x0
	fcvt	h0, s0
	ldr	x9, [sp, #128]                  ; 8-byte Folded Reload
	mov	x10, x24
LBB0_41:                                ;   Parent Loop BB0_4 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q1, [x10], #16
	ldr	q2, [x9]
	fmla.8h	v2, v1, v0[0]
	str	q2, [x9], #16
	add	x8, x8, #8
	cmp	x8, x27
	b.lo	LBB0_41
; %bb.42:                               ;   in Loop: Header=BB0_11 Depth=2
	add	x21, x21, #1
	cmp	x21, x23
	b.ne	LBB0_10
	b	LBB0_44
LBB0_43:                                ;   in Loop: Header=BB0_4 Depth=1
	cmp	w22, #1
	b.lt	LBB0_3
LBB0_44:                                ;   in Loop: Header=BB0_4 Depth=1
	mov	x8, #0                          ; =0x0
	ldr	x9, [sp, #64]                   ; 8-byte Folded Reload
	ldr	x10, [sp, #128]                 ; 8-byte Folded Reload
	fdiv	s0, s9, s10
LBB0_45:                                ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q1, [x10], #16
	fcvtl	v2.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v2, v2, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q2, q1, [x9, #-16]
	add	x8, x8, #8
	add	x9, x9, #32
	cmp	x8, x26
	b.lt	LBB0_45
	b	LBB0_3
LBB0_46:
	cmp	w22, #0
	b.le	LBB0_53
; %bb.47:
	mov	x19, #0                         ; =0x0
	mov	w20, w4
	add	x21, x0, #16
	lsl	x22, x27, #2
	ldr	x8, [sp, #8]                    ; 8-byte Folded Reload
	add	x23, x8, #16
	mvni.4s	v0, #127, msl #16
	fneg.4s	v2, v0
	str	q2, [sp, #112]                  ; 16-byte Folded Spill
LBB0_48:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_49 Depth 2
                                        ;     Child Loop BB0_51 Depth 2
	mov	x8, #0                          ; =0x0
	mov	x9, x25
	mov	x10, x21
LBB0_49:                                ;   Parent Loop BB0_48 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x10, #-16]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	str	q0, [x9], #16
	add	x8, x8, #8
	add	x10, x10, #32
	cmp	x8, x26
	b.lt	LBB0_49
; %bb.50:                               ;   in Loop: Header=BB0_48 Depth=1
	ldr	x24, [sp, #128]                 ; 8-byte Folded Reload
	mov	x0, x24
	ldr	x1, [sp, #32]                   ; 8-byte Folded Reload
	bl	_bzero
	mov	x8, #0                          ; =0x0
	mov	x9, x23
	ldr	q2, [sp, #112]                  ; 16-byte Folded Reload
LBB0_51:                                ;   Parent Loop BB0_48 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q0, [x24], #16
	fcvtl	v1.4s, v0.4h
	fcvtl2	v0.4s, v0.8h
	fmul.4s	v1, v1, v2
	fmul.4s	v0, v0, v2
	stp	q1, q0, [x9, #-16]
	add	x8, x8, #8
	add	x9, x9, #32
	cmp	x8, x26
	b.lt	LBB0_51
; %bb.52:                               ;   in Loop: Header=BB0_48 Depth=1
	add	x19, x19, #1
	add	x21, x21, x22
	add	x23, x23, x22
	cmp	x19, x20
	b.ne	LBB0_48
	b	LBB0_55
LBB0_53:
	ldr	x0, [sp, #128]                  ; 8-byte Folded Reload
	ldr	x1, [sp, #32]                   ; 8-byte Folded Reload
	mov	x19, x4
	bl	_bzero
	mov	x8, x19
LBB0_54:                                ; =>This Inner Loop Header: Depth=1
	subs	w8, w8, #1
	b.ne	LBB0_54
LBB0_55:
	ldr	x8, [sp, #8]                    ; 8-byte Folded Reload
	ldr	s0, [x8]
	adrp	x8, _dummy_result@PAGE
	str	s0, [x8, _dummy_result@PAGEOFF]
	ldp	x29, x30, [sp, #256]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #240]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #224]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #208]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #192]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #176]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #160]              ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #144]            ; 16-byte Folded Reload
	add	sp, sp, #272
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	stp	d15, d14, [sp, #-160]!          ; 16-byte Folded Spill
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #80]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #96]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #112]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #128]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #144]            ; 16-byte Folded Spill
	add	x29, sp, #144
	sub	sp, sp, #656
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
	.cfi_offset b14, -152
	.cfi_offset b15, -160
	mov	x9, #0                          ; =0x0
	mov	x8, #70368744177664             ; =0x400000000000
	movk	x8, #16527, lsl #48
	fmov	d8, x8
	mov	w26, #16576                     ; =0x40c0
	mov	w27, #16512                     ; =0x4080
	mov	w19, #16448                     ; =0x4040
	mov	w28, #1267                      ; =0x4f3
	movk	w28, #15797, lsl #16
	fmov	s9, #1.00000000
	fmov	d10, #20.00000000
	b	LBB1_2
LBB1_1:                                 ;   in Loop: Header=BB1_2 Depth=1
	sub	x1, x29, #184
	mov	w0, #6                          ; =0x6
	bl	_clock_gettime
	ldp	x8, x9, [x29, #-184]
	ldp	x10, x11, [x29, #-168]
	sub	x8, x8, x10
	scvtf	d0, x8
	sub	x8, x9, x11
	scvtf	d1, x8
	mov	x8, #225833675390976            ; =0xcd6500000000
	movk	x8, #16845, lsl #48
	fmov	d2, x8
	fmadd	d0, d0, d2, d1
	fdiv	d0, d0, d10
	mov	x8, #145685290680320            ; =0x848000000000
	movk	x8, #16686, lsl #48
	fmov	d1, x8
	fdiv	d1, d0, d1
	ldr	x9, [sp, #32]                   ; 8-byte Folded Reload
	scvtf	d3, w9
	mov	x8, #4634204016564240384        ; =0x4050000000000000
	fmov	d4, x8
	fmul	d3, d3, d4
	mov	x8, #4638707616191610880        ; =0x4060000000000000
	fmov	d4, x8
	fmul	d3, d3, d4
	fdiv	d0, d3, d0
	fmul	d0, d0, d2
	fdiv	d0, d0, d2
	str	x9, [sp]
	stp	d1, d0, [sp, #8]
Lloh0:
	adrp	x0, l_.str@PAGE
Lloh1:
	add	x0, x0, l_.str@PAGEOFF
	bl	_printf
	mov	x0, x20
	bl	_free
	mov	x0, x21
	bl	_free
	mov	x0, x22
	bl	_free
	ldr	x0, [sp, #88]                   ; 8-byte Folded Reload
	bl	_free
	mov	x0, x24
	bl	_free
	mov	x0, x25
	bl	_free
	ldr	x9, [sp, #24]                   ; 8-byte Folded Reload
	add	x9, x9, #1
	cmp	x9, #5
	b.eq	LBB1_24
LBB1_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_3 Depth 2
                                        ;     Child Loop BB1_5 Depth 2
                                        ;     Child Loop BB1_8 Depth 2
                                        ;       Child Loop BB1_10 Depth 3
                                        ;         Child Loop BB1_13 Depth 4
                                        ;     Child Loop BB1_17 Depth 2
                                        ;       Child Loop BB1_19 Depth 3
                                        ;         Child Loop BB1_22 Depth 4
Lloh2:
	adrp	x8, l___const.main.seq_lens@PAGE
Lloh3:
	add	x8, x8, l___const.main.seq_lens@PAGEOFF
	ldrsw	x8, [x8, x9, lsl #2]
	stp	x9, x8, [sp, #24]               ; 16-byte Folded Spill
	lsl	x22, x8, #13
	mov	w0, #64                         ; =0x40
	mov	w1, #16384                      ; =0x4000
	bl	_aligned_alloc
	mov	x23, x0
	mov	w0, #64                         ; =0x40
	mov	x1, x22
	bl	_aligned_alloc
	mov	x21, x0
	mov	w0, #64                         ; =0x40
	mov	x1, x22
	bl	_aligned_alloc
	mov	x22, x0
	mov	w0, #64                         ; =0x40
	mov	w1, #16384                      ; =0x4000
	bl	_aligned_alloc
	str	x0, [sp, #88]                   ; 8-byte Folded Spill
	mov	w0, #64                         ; =0x40
	mov	w1, #256                        ; =0x100
	bl	_aligned_alloc
	mov	x24, x0
	mov	w0, #64                         ; =0x40
	mov	w1, #256                        ; =0x100
	bl	_aligned_alloc
	mov	x25, x0
	mov	x20, #0                         ; =0x0
LBB1_3:                                 ;   Parent Loop BB1_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	bl	_rand
	mov	w8, #19923                      ; =0x4dd3
	movk	w8, #4194, lsl #16
	smull	x8, w0, w8
	asr	x8, x8, #39
	add	w8, w8, w8, lsr #31
	mov	w9, #2000                       ; =0x7d0
	msub	w8, w8, w9, w0
	sub	w8, w8, #1000
	scvtf	d0, w8
	fdiv	d0, d0, d8
	fcvt	s0, d0
	str	s0, [x23, x20]
	add	x20, x20, #4
	cmp	x20, #4, lsl #12                ; =16384
	b.ne	LBB1_3
; %bb.4:                                ;   in Loop: Header=BB1_2 Depth=1
	str	x23, [sp, #80]                  ; 8-byte Folded Spill
	mov	x20, #0                         ; =0x0
	ldr	x8, [sp, #32]                   ; 8-byte Folded Reload
	lsl	x8, x8, #12
	cmp	x8, #1
	csinc	x8, x8, xzr, hi
	stur	x8, [x29, #-208]                ; 8-byte Folded Spill
LBB1_5:                                 ;   Parent Loop BB1_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	bl	_rand
	mov	w23, #19923                     ; =0x4dd3
	movk	w23, #4194, lsl #16
	smull	x8, w0, w23
	asr	x8, x8, #38
	add	w8, w8, w8, lsr #31
	mov	w9, #1000                       ; =0x3e8
	msub	w8, w8, w9, w0
	scvtf	d0, w8
	fdiv	d0, d0, d8
	fcvtzs	w8, d0
	strh	w8, [x21, x20, lsl #1]
	bl	_rand
	smull	x8, w0, w23
	asr	x8, x8, #38
	add	w8, w8, w8, lsr #31
	mov	w9, #1000                       ; =0x3e8
	msub	w8, w8, w9, w0
	scvtf	d0, w8
	fdiv	d0, d0, d8
	fcvtzs	w8, d0
	strh	w8, [x22, x20, lsl #1]
	add	x20, x20, #1
	ldur	x8, [x29, #-208]                ; 8-byte Folded Reload
	cmp	x8, x20
	b.ne	LBB1_5
; %bb.6:                                ;   in Loop: Header=BB1_2 Depth=1
	mov	w9, #0                          ; =0x0
	ldr	x8, [sp, #32]                   ; 8-byte Folded Reload
	mov	w8, w8
	str	x8, [sp, #72]                   ; 8-byte Folded Spill
	ldr	x20, [sp, #80]                  ; 8-byte Folded Reload
	b	LBB1_8
LBB1_7:                                 ;   in Loop: Header=BB1_8 Depth=2
	ldr	x8, [sp, #88]                   ; 8-byte Folded Reload
	ldr	s0, [x8]
	adrp	x8, _dummy_result@PAGE
	str	s0, [x8, _dummy_result@PAGEOFF]
	ldr	w9, [sp, #44]                   ; 4-byte Folded Reload
	add	w9, w9, #1
	cmp	w9, #5
	b.eq	LBB1_15
LBB1_8:                                 ;   Parent Loop BB1_2 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_10 Depth 3
                                        ;         Child Loop BB1_13 Depth 4
	str	w9, [sp, #44]                   ; 4-byte Folded Spill
	mov	x9, #0                          ; =0x0
	mov	x10, #0                         ; =0x0
	b	LBB1_10
LBB1_9:                                 ;   in Loop: Header=BB1_10 Depth=3
	stp	q22, q16, [x24, #32]
	stp	q21, q20, [x24, #64]
	stp	q17, q24, [x24, #128]
	fdiv	s0, s9, s11
	fcvtl	v1.4s, v23.4h
	ext.16b	v2, v23, v23, #8
	stp	q19, q18, [x24, #96]
	fcvtl	v2.4s, v2.4h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q3, q4, [x24, #160]
	fcvtl	v3.4s, v7.4h
	ext.16b	v4, v7, v7, #8
	fcvtl	v4.4s, v4.4h
	stp	q5, q26, [x24, #192]
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v4, v4, v0[0]
	fcvtl	v5.4s, v22.4h
	stp	q6, q25, [x24, #224]
	ext.16b	v6, v22, v22, #8
	fcvtl	v6.4s, v6.4h
	fmul.4s	v5, v5, v0[0]
	stp	q23, q7, [x24]
	ldr	x8, [sp, #88]                   ; 8-byte Folded Reload
	ldr	x9, [sp, #48]                   ; 8-byte Folded Reload
	add	x8, x8, x9, lsl #2
	fmul.4s	v6, v6, v0[0]
	fcvtl	v7.4s, v16.4h
	ext.16b	v16, v16, v16, #8
	stp	q1, q2, [x8]
	fcvtl	v1.4s, v16.4h
	fmul.4s	v2, v7, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q3, q4, [x8, #32]
	fcvtl	v3.4s, v21.4h
	ext.16b	v4, v21, v21, #8
	fcvtl	v4.4s, v4.4h
	stp	q5, q6, [x8, #64]
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v4, v4, v0[0]
	fcvtl	v5.4s, v20.4h
	stp	q2, q1, [x8, #96]
	ext.16b	v1, v20, v20, #8
	fcvtl	v1.4s, v1.4h
	fmul.4s	v2, v5, v0[0]
	stp	q3, q4, [x8, #128]
	fmul.4s	v1, v1, v0[0]
	fcvtl	v3.4s, v19.4h
	ext.16b	v4, v19, v19, #8
	stp	q2, q1, [x8, #160]
	fcvtl	v1.4s, v4.4h
	fmul.4s	v2, v3, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q2, q1, [x8, #192]
	fcvtl	v1.4s, v18.4h
	ext.16b	v2, v18, v18, #8
	fcvtl	v2.4s, v2.4h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #224]
	fcvtl	v1.4s, v17.4h
	ext.16b	v2, v17, v17, #8
	fcvtl	v2.4s, v2.4h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #256]
	ldp	q1, q2, [x24, #144]
	fcvtl	v3.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q3, q1, [x8, #288]
	fcvtl	v1.4s, v2.4h
	fcvtl2	v2.4s, v2.8h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #320]
	ldp	q1, q2, [x24, #176]
	fcvtl	v3.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q3, q1, [x8, #352]
	fcvtl	v1.4s, v2.4h
	fcvtl2	v2.4s, v2.8h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	ldp	q3, q4, [x24, #208]
	fcvtl	v5.4s, v3.4h
	stp	q1, q2, [x8, #384]
	fcvtl2	v1.4s, v3.8h
	fmul.4s	v2, v5, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q2, q1, [x8, #416]
	fcvtl	v1.4s, v4.4h
	fcvtl2	v2.4s, v4.8h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #448]
	ldr	q1, [x24, #240]
	fcvtl	v2.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v2, v2, v0[0]
	fmul.4s	v0, v1, v0[0]
	ldp	x10, x9, [sp, #56]              ; 16-byte Folded Reload
	add	x10, x10, #1
	add	x9, x9, #256
	stp	q2, q0, [x8, #480]
	cmp	x10, #32
	ldr	x20, [sp, #80]                  ; 8-byte Folded Reload
	b.eq	LBB1_7
LBB1_10:                                ;   Parent Loop BB1_2 Depth=1
                                        ;     Parent Loop BB1_8 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB1_13 Depth 4
	add	x8, x20, x10, lsl #9
	ldp	q0, q1, [x8]
	str	x10, [sp, #56]                  ; 8-byte Folded Spill
	lsl	x10, x10, #7
	str	x10, [sp, #48]                  ; 8-byte Folded Spill
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #32]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25]
	ldp	q0, q1, [x8, #64]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #96]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #32]
	ldp	q0, q1, [x8, #128]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #160]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #64]
	ldp	q0, q1, [x8, #192]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #224]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #96]
	ldp	q0, q1, [x8, #256]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #288]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #128]
	ldp	q0, q1, [x8, #320]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #352]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #160]
	ldp	q0, q1, [x8, #384]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #416]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #192]
	ldp	q0, q1, [x8, #448]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #480]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #224]
	movi.2d	v0, #0000000000000000
	stp	q0, q0, [x24, #224]
	stp	q0, q0, [x24, #192]
	stp	q0, q0, [x24, #160]
	stp	q0, q0, [x24, #128]
	stp	q0, q0, [x24, #96]
	stp	q0, q0, [x24, #64]
	stp	q0, q0, [x24, #32]
	stp	q0, q0, [x24]
	ldp	q2, q3, [x25]
	stp	q2, q1, [sp, #400]              ; 32-byte Folded Spill
	ldp	q2, q4, [x25, #32]
	stp	q2, q3, [sp, #368]              ; 32-byte Folded Spill
	ldp	q2, q3, [x25, #64]
	stp	q2, q4, [sp, #336]              ; 32-byte Folded Spill
	ldp	q2, q4, [x25, #96]
	stp	q2, q3, [sp, #304]              ; 32-byte Folded Spill
	ldp	q2, q3, [x25, #128]
	stp	q2, q4, [sp, #272]              ; 32-byte Folded Spill
	ldp	q2, q4, [x25, #160]
	stp	q2, q3, [sp, #240]              ; 32-byte Folded Spill
	ldp	q2, q3, [x25, #192]
	stp	q2, q4, [sp, #208]              ; 32-byte Folded Spill
	ldr	q0, [x25, #224]
	stp	q0, q3, [sp, #176]              ; 32-byte Folded Spill
	ldp	q7, q22, [x24, #16]
	ldp	q16, q21, [x24, #48]
	ldp	q20, q19, [x24, #80]
	ldp	q18, q17, [x24, #112]
	ldp	q24, q3, [x24, #144]
	movi.2d	v11, #0000000000000000
	movi.2d	v27, #0000000000000000
	mov	w8, #-8388608                   ; =0xff800000
	fmov	s0, w8
	ldr	x23, [sp, #72]                  ; 8-byte Folded Reload
	str	x9, [sp, #64]                   ; 8-byte Folded Spill
	mov	x20, x9
	ldp	q4, q5, [x24, #176]
	movi.2d	v28, #0000000000000000
	movi.2d	v29, #0000000000000000
	movi.2d	v30, #0000000000000000
	ldp	q26, q6, [x24, #208]
	movi.2d	v31, #0000000000000000
	movi.2d	v23, #0000000000000000
	ldr	q25, [x24, #240]
	b	LBB1_13
LBB1_11:                                ;   in Loop: Header=BB1_13 Depth=4
	fsub	s0, s0, s1
	stp	q28, q27, [sp, #144]            ; 32-byte Folded Spill
	stp	q30, q29, [sp, #112]            ; 32-byte Folded Spill
	str	q31, [sp, #96]                  ; 16-byte Folded Spill
	bl	_expf
	fcvt	h1, s0
	ldr	q23, [sp, #496]                 ; 16-byte Folded Reload
	fmul.8h	v23, v23, v1[0]
	ldp	q3, q2, [sp, #96]               ; 32-byte Folded Reload
	fmul.8h	v7, v3, v1[0]
	fmul.8h	v22, v2, v1[0]
	ldp	q3, q2, [sp, #128]              ; 32-byte Folded Reload
	fmul.8h	v16, v3, v1[0]
	fmul.8h	v21, v2, v1[0]
	ldr	q2, [sp, #160]                  ; 16-byte Folded Reload
	fmul.8h	v20, v2, v1[0]
	ldp	q19, q18, [x29, #-240]          ; 32-byte Folded Reload
	fmul.8h	v19, v19, v1[0]
	fmul.8h	v18, v18, v1[0]
	ldur	q17, [x29, #-208]               ; 16-byte Folded Reload
	fmul.8h	v17, v17, v1[0]
	ldp	q24, q6, [sp, #464]             ; 32-byte Folded Reload
	fmul.8h	v24, v24, v1[0]
	ldur	q3, [x29, #-256]                ; 16-byte Folded Reload
	fmul.8h	v3, v3, v1[0]
	ldp	q4, q5, [sp, #512]              ; 32-byte Folded Reload
	fmul.8h	v4, v4, v1[0]
	fmul.8h	v5, v5, v1[0]
	ldp	q26, q25, [sp, #432]            ; 32-byte Folded Reload
	fmul.8h	v26, v26, v1[0]
	fmul.8h	v6, v6, v1[0]
	fmul.8h	v25, v25, v1[0]
	fmul	s0, s11, s0
	fadd	s11, s0, s9
	fmov	h0, #1.00000000
LBB1_12:                                ;   in Loop: Header=BB1_13 Depth=4
	add	x8, x22, x20
	add	x9, x8, x26
	add	x10, x8, #4, lsl #12            ; =16384
	prfm	pldl1keep, [x10]
	add	x10, x8, x27
	add	x11, x22, x20
	ldp	q1, q2, [x11]
	add	x8, x8, x19
	fmla.8h	v23, v1, v0[0]
	fmla.8h	v7, v2, v0[0]
	ldp	q1, q2, [x11, #32]
	fmla.8h	v22, v1, v0[0]
	fmla.8h	v16, v2, v0[0]
	prfm	pldl1keep, [x8]
	ldp	q1, q2, [x11, #64]
	fmla.8h	v21, v1, v0[0]
	fmla.8h	v20, v2, v0[0]
	ldp	q1, q2, [x11, #96]
	fmla.8h	v19, v1, v0[0]
	fmla.8h	v18, v2, v0[0]
	prfm	pldl1keep, [x10]
	ldp	q1, q2, [x11, #128]
	fmla.8h	v17, v1, v0[0]
	fmla.8h	v24, v2, v0[0]
	ldp	q1, q2, [x11, #160]
	fmla.8h	v3, v1, v0[0]
	fmla.8h	v4, v2, v0[0]
	prfm	pldl1keep, [x9]
	ldp	q1, q2, [x11, #192]
	fmla.8h	v5, v1, v0[0]
	fmla.8h	v26, v2, v0[0]
	ldp	q1, q2, [x11, #224]
	fmla.8h	v6, v1, v0[0]
	add	x20, x20, #2, lsl #12           ; =8192
	mov.16b	v27, v20
	mov.16b	v28, v21
	mov.16b	v29, v16
	fmla.8h	v25, v2, v0[0]
	mov.16b	v30, v22
	mov.16b	v31, v7
	mov.16b	v0, v12
	subs	x23, x23, #1
	b.eq	LBB1_9
LBB1_13:                                ;   Parent Loop BB1_2 Depth=1
                                        ;     Parent Loop BB1_8 Depth=2
                                        ;       Parent Loop BB1_10 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	stp	q26, q25, [sp, #432]            ; 32-byte Folded Spill
	stp	q24, q6, [sp, #464]             ; 32-byte Folded Spill
	stp	q23, q4, [sp, #496]             ; 32-byte Folded Spill
	str	q5, [sp, #528]                  ; 16-byte Folded Spill
	stp	q3, q19, [x29, #-256]           ; 32-byte Folded Spill
	stp	q18, q17, [x29, #-224]          ; 32-byte Folded Spill
	add	x8, x21, x20
	prfm	pldl1keep, [x8, #16384]
	movi.2d	v1, #0000000000000000
	ldp	q2, q3, [x8]
	ldp	q4, q5, [sp, #384]              ; 32-byte Folded Reload
	fmla.8h	v1, v2, v5
	movi.2d	v2, #0000000000000000
	fmla.8h	v2, v3, v4
	movi.2d	v3, #0000000000000000
	ldp	q4, q5, [x8, #32]
	ldr	q6, [sp, #368]                  ; 16-byte Folded Reload
	fmla.8h	v3, v4, v6
	movi.2d	v4, #0000000000000000
	ldp	q17, q6, [sp, #336]             ; 32-byte Folded Reload
	fmla.8h	v4, v5, v6
	prfm	pldl1keep, [x8, #16448]
	ldp	q5, q6, [x8, #64]
	fmla.8h	v1, v5, v17
	ldp	q17, q5, [sp, #304]             ; 32-byte Folded Reload
	fmla.8h	v2, v6, v5
	ldp	q5, q6, [x8, #96]
	fmla.8h	v3, v5, v17
	ldp	q17, q5, [sp, #272]             ; 32-byte Folded Reload
	fmla.8h	v4, v6, v5
	prfm	pldl1keep, [x8, #16512]
	ldp	q5, q6, [x8, #128]
	fmla.8h	v1, v5, v17
	ldp	q17, q5, [sp, #240]             ; 32-byte Folded Reload
	fmla.8h	v2, v6, v5
	ldp	q5, q6, [x8, #160]
	fmla.8h	v3, v5, v17
	ldp	q17, q5, [sp, #208]             ; 32-byte Folded Reload
	fmla.8h	v4, v6, v5
	prfm	pldl1keep, [x8, #16576]
	ldp	q5, q6, [x8, #192]
	fmla.8h	v1, v5, v17
	ldp	q17, q5, [sp, #176]             ; 32-byte Folded Reload
	fmla.8h	v2, v6, v5
	ldp	q5, q6, [x8, #224]
	fmla.8h	v3, v5, v17
	ldr	q5, [sp, #416]                  ; 16-byte Folded Reload
	fmla.8h	v4, v6, v5
	fadd.8h	v1, v1, v2
	fadd.8h	v2, v3, v4
	fadd.8h	v1, v1, v2
	fcvtl	v2.4s, v1.4h
	ext.16b	v1, v1, v1, #8
	fcvtl	v1.4s, v1.4h
	fadd.4s	v1, v2, v1
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	fmov	s2, w28
	fmul	s1, s1, s2
	fcmp	s1, s0
	fcsel	s12, s1, s0, gt
	b.gt	LBB1_11
; %bb.14:                               ;   in Loop: Header=BB1_13 Depth=4
	fsub	s0, s1, s0
	stp	q21, q20, [sp, #144]            ; 32-byte Folded Spill
	stp	q7, q16, [sp, #112]             ; 32-byte Folded Spill
	str	q22, [sp, #96]                  ; 16-byte Folded Spill
	bl	_expf
	ldp	q22, q7, [sp, #96]              ; 32-byte Folded Reload
	ldp	q16, q21, [sp, #128]            ; 32-byte Folded Reload
	ldr	q20, [sp, #160]                 ; 16-byte Folded Reload
	fadd	s11, s11, s0
	fcvt	h0, s0
	ldp	q18, q17, [x29, #-224]          ; 32-byte Folded Reload
	ldp	q3, q19, [x29, #-256]           ; 32-byte Folded Reload
	ldp	q4, q5, [sp, #512]              ; 32-byte Folded Reload
	ldp	q6, q23, [sp, #480]             ; 32-byte Folded Reload
	ldp	q25, q24, [sp, #448]            ; 32-byte Folded Reload
	ldr	q26, [sp, #432]                 ; 16-byte Folded Reload
	b	LBB1_12
LBB1_15:                                ;   in Loop: Header=BB1_2 Depth=1
	sub	x1, x29, #168
	mov	w0, #6                          ; =0x6
	bl	_clock_gettime
	mov	w8, #0                          ; =0x0
	b	LBB1_17
LBB1_16:                                ;   in Loop: Header=BB1_17 Depth=2
	ldr	x8, [sp, #88]                   ; 8-byte Folded Reload
	ldr	s0, [x8]
	adrp	x8, _dummy_result@PAGE
	str	s0, [x8, _dummy_result@PAGEOFF]
	ldr	w8, [sp, #44]                   ; 4-byte Folded Reload
	add	w8, w8, #1
	cmp	w8, #20
	b.eq	LBB1_1
LBB1_17:                                ;   Parent Loop BB1_2 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_19 Depth 3
                                        ;         Child Loop BB1_22 Depth 4
	str	w8, [sp, #44]                   ; 4-byte Folded Spill
	mov	x9, #0                          ; =0x0
	mov	x10, #0                         ; =0x0
	b	LBB1_19
LBB1_18:                                ;   in Loop: Header=BB1_19 Depth=3
	stp	q23, q16, [x24, #32]
	stp	q18, q20, [x24, #64]
	stp	q19, q31, [x24, #128]
	fdiv	s0, s9, s11
	fcvtl	v1.4s, v24.4h
	ext.16b	v2, v24, v24, #8
	stp	q17, q21, [x24, #96]
	fcvtl	v2.4s, v2.4h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q28, q30, [x24, #160]
	fcvtl	v3.4s, v22.4h
	ext.16b	v4, v22, v22, #8
	fcvtl	v4.4s, v4.4h
	stp	q27, q29, [x24, #192]
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v4, v4, v0[0]
	fcvtl	v5.4s, v23.4h
	stp	q26, q25, [x24, #224]
	ext.16b	v6, v23, v23, #8
	fcvtl	v6.4s, v6.4h
	fmul.4s	v5, v5, v0[0]
	stp	q24, q22, [x24]
	ldr	x8, [sp, #88]                   ; 8-byte Folded Reload
	ldr	x9, [sp, #48]                   ; 8-byte Folded Reload
	add	x8, x8, x9, lsl #2
	fmul.4s	v6, v6, v0[0]
	fcvtl	v7.4s, v16.4h
	ext.16b	v16, v16, v16, #8
	stp	q1, q2, [x8]
	fcvtl	v1.4s, v16.4h
	fmul.4s	v2, v7, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q3, q4, [x8, #32]
	fcvtl	v3.4s, v18.4h
	ext.16b	v4, v18, v18, #8
	fcvtl	v4.4s, v4.4h
	stp	q5, q6, [x8, #64]
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v4, v4, v0[0]
	fcvtl	v5.4s, v20.4h
	stp	q2, q1, [x8, #96]
	ext.16b	v1, v20, v20, #8
	fcvtl	v1.4s, v1.4h
	fmul.4s	v2, v5, v0[0]
	stp	q3, q4, [x8, #128]
	fmul.4s	v1, v1, v0[0]
	fcvtl	v3.4s, v17.4h
	ext.16b	v4, v17, v17, #8
	stp	q2, q1, [x8, #160]
	fcvtl	v1.4s, v4.4h
	fmul.4s	v2, v3, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q2, q1, [x8, #192]
	fcvtl	v1.4s, v21.4h
	ext.16b	v2, v21, v21, #8
	fcvtl	v2.4s, v2.4h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #224]
	fcvtl	v1.4s, v19.4h
	ext.16b	v2, v19, v19, #8
	fcvtl	v2.4s, v2.4h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #256]
	ldp	q1, q2, [x24, #144]
	fcvtl	v3.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q3, q1, [x8, #288]
	fcvtl	v1.4s, v2.4h
	fcvtl2	v2.4s, v2.8h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #320]
	ldp	q1, q2, [x24, #176]
	fcvtl	v3.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v3, v3, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q3, q1, [x8, #352]
	fcvtl	v1.4s, v2.4h
	fcvtl2	v2.4s, v2.8h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	ldp	q3, q4, [x24, #208]
	fcvtl	v5.4s, v3.4h
	stp	q1, q2, [x8, #384]
	fcvtl2	v1.4s, v3.8h
	fmul.4s	v2, v5, v0[0]
	fmul.4s	v1, v1, v0[0]
	stp	q2, q1, [x8, #416]
	fcvtl	v1.4s, v4.4h
	fcvtl2	v2.4s, v4.8h
	fmul.4s	v1, v1, v0[0]
	fmul.4s	v2, v2, v0[0]
	stp	q1, q2, [x8, #448]
	ldr	q1, [x24, #240]
	fcvtl	v2.4s, v1.4h
	fcvtl2	v1.4s, v1.8h
	fmul.4s	v2, v2, v0[0]
	fmul.4s	v0, v1, v0[0]
	ldp	x10, x9, [sp, #56]              ; 16-byte Folded Reload
	add	x10, x10, #1
	add	x9, x9, #256
	stp	q2, q0, [x8, #480]
	cmp	x10, #32
	ldr	x20, [sp, #80]                  ; 8-byte Folded Reload
	b.eq	LBB1_16
LBB1_19:                                ;   Parent Loop BB1_2 Depth=1
                                        ;     Parent Loop BB1_17 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB1_22 Depth 4
	add	x8, x20, x10, lsl #9
	ldp	q0, q1, [x8]
	str	x10, [sp, #56]                  ; 8-byte Folded Spill
	lsl	x10, x10, #7
	str	x10, [sp, #48]                  ; 8-byte Folded Spill
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #32]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25]
	ldp	q0, q1, [x8, #64]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #96]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #32]
	ldp	q0, q1, [x8, #128]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #160]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #64]
	ldp	q0, q1, [x8, #192]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #224]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #96]
	ldp	q0, q1, [x8, #256]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #288]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #128]
	ldp	q0, q1, [x8, #320]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #352]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #160]
	ldp	q0, q1, [x8, #384]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #416]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #192]
	ldp	q0, q1, [x8, #448]
	fcvtn	v0.4h, v0.4s
	fcvtn2	v0.8h, v1.4s
	ldp	q1, q2, [x8, #480]
	fcvtn	v1.4h, v1.4s
	fcvtn2	v1.8h, v2.4s
	stp	q0, q1, [x25, #224]
	movi.2d	v0, #0000000000000000
	stp	q0, q0, [x24, #224]
	stp	q0, q0, [x24, #192]
	stp	q0, q0, [x24, #160]
	stp	q0, q0, [x24, #128]
	stp	q0, q0, [x24, #96]
	stp	q0, q0, [x24, #64]
	stp	q0, q0, [x24, #32]
	stp	q0, q0, [x24]
	ldp	q2, q3, [x25]
	stp	q2, q1, [sp, #368]              ; 32-byte Folded Spill
	ldp	q2, q4, [x25, #32]
	stp	q2, q3, [sp, #336]              ; 32-byte Folded Spill
	ldp	q2, q3, [x25, #64]
	stp	q2, q4, [sp, #304]              ; 32-byte Folded Spill
	ldp	q2, q4, [x25, #96]
	stp	q2, q3, [sp, #272]              ; 32-byte Folded Spill
	ldp	q2, q3, [x25, #128]
	stp	q2, q4, [sp, #240]              ; 32-byte Folded Spill
	ldp	q2, q4, [x25, #160]
	stp	q2, q3, [sp, #208]              ; 32-byte Folded Spill
	ldp	q2, q3, [x25, #192]
	stp	q2, q4, [sp, #176]              ; 32-byte Folded Spill
	ldr	q0, [x25, #224]
	stp	q0, q3, [sp, #144]              ; 32-byte Folded Spill
	ldp	q22, q23, [x24, #16]
	ldp	q16, q18, [x24, #48]
	movi.2d	v11, #0000000000000000
	movi.2d	v13, #0000000000000000
	mov	w8, #-8388608                   ; =0xff800000
	fmov	s0, w8
	ldr	x23, [sp, #72]                  ; 8-byte Folded Reload
	str	x9, [sp, #64]                   ; 8-byte Folded Spill
	mov	x20, x9
	ldp	q20, q17, [x24, #80]
	movi.2d	v14, #0000000000000000
	movi.2d	v15, #0000000000000000
	movi.2d	v6, #0000000000000000
	ldp	q21, q19, [x24, #112]
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
	ldp	q31, q28, [x24, #144]
	movi.2d	v5, #0000000000000000
	movi.2d	v1, #0000000000000000
	movi.2d	v7, #0000000000000000
	stp	q7, q7, [x29, #-256]            ; 32-byte Folded Spill
	ldp	q30, q27, [x24, #176]
	ldp	q29, q26, [x24, #208]
	movi.2d	v24, #0000000000000000
	stp	q24, q7, [sp, #512]             ; 32-byte Folded Spill
	ldr	q25, [x24, #240]
	b	LBB1_22
LBB1_20:                                ;   in Loop: Header=BB1_22 Depth=4
	fsub	s0, s0, s1
	stp	q14, q13, [sp, #112]            ; 32-byte Folded Spill
	str	q15, [sp, #96]                  ; 16-byte Folded Spill
	bl	_expf
	fcvt	h1, s0
	ldur	q24, [x29, #-208]               ; 16-byte Folded Reload
	fmul.8h	v24, v24, v1[0]
	ldr	q2, [sp, #512]                  ; 16-byte Folded Reload
	fmul.8h	v22, v2, v1[0]
	ldr	q2, [sp, #400]                  ; 16-byte Folded Reload
	fmul.8h	v23, v2, v1[0]
	ldr	q2, [sp, #528]                  ; 16-byte Folded Reload
	fmul.8h	v16, v2, v1[0]
	ldp	q3, q2, [x29, #-256]            ; 32-byte Folded Reload
	fmul.8h	v18, v3, v1[0]
	fmul.8h	v20, v2, v1[0]
	ldp	q3, q2, [sp, #416]              ; 32-byte Folded Reload
	fmul.8h	v17, v3, v1[0]
	fmul.8h	v21, v2, v1[0]
	ldp	q3, q2, [sp, #448]              ; 32-byte Folded Reload
	fmul.8h	v19, v3, v1[0]
	fmul.8h	v31, v2, v1[0]
	ldp	q3, q2, [sp, #480]              ; 32-byte Folded Reload
	fmul.8h	v28, v3, v1[0]
	fmul.8h	v30, v2, v1[0]
	ldp	q3, q2, [sp, #96]               ; 32-byte Folded Reload
	fmul.8h	v27, v3, v1[0]
	fmul.8h	v29, v2, v1[0]
	ldr	q2, [sp, #128]                  ; 16-byte Folded Reload
	fmul.8h	v26, v2, v1[0]
	ldur	q25, [x29, #-224]               ; 16-byte Folded Reload
	fmul.8h	v25, v25, v1[0]
	fmul	s0, s11, s0
	fadd	s11, s0, s9
	fmov	h0, #1.00000000
LBB1_21:                                ;   in Loop: Header=BB1_22 Depth=4
	add	x8, x22, x20
	add	x9, x8, x26
	add	x10, x8, x27
	add	x11, x8, #4, lsl #12            ; =16384
	prfm	pldl1keep, [x11]
	add	x11, x8, x19
	add	x8, x22, x20
	ldp	q1, q2, [x8]
	fmla.8h	v24, v1, v0[0]
	fmla.8h	v22, v2, v0[0]
	ldp	q1, q2, [x8, #32]
	fmla.8h	v23, v1, v0[0]
	fmla.8h	v16, v2, v0[0]
	prfm	pldl1keep, [x11]
	ldp	q1, q2, [x8, #64]
	fmla.8h	v18, v1, v0[0]
	fmla.8h	v20, v2, v0[0]
	ldp	q1, q2, [x8, #96]
	prfm	pldl1keep, [x10]
	ldp	q3, q4, [x8, #160]
	fmla.8h	v30, v4, v0[0]
	ldp	q4, q5, [x8, #128]
	prfm	pldl1keep, [x9]
	ldp	q6, q7, [x8, #192]
	fmla.8h	v27, v6, v0[0]
	fmla.8h	v29, v7, v0[0]
	add	x20, x20, #2, lsl #12           ; =8192
	ldp	q6, q7, [x8, #224]
	fmla.8h	v26, v6, v0[0]
	mov.16b	v13, v26
	mov.16b	v14, v29
	mov.16b	v15, v27
	mov.16b	v6, v30
	fmla.8h	v21, v2, v0[0]
	fmla.8h	v19, v4, v0[0]
	fmla.8h	v31, v5, v0[0]
	fmla.8h	v28, v3, v0[0]
	mov.16b	v2, v28
	mov.16b	v3, v31
	mov.16b	v4, v19
	mov.16b	v5, v21
	fmla.8h	v17, v1, v0[0]
	mov.16b	v1, v17
	stp	q18, q20, [x29, #-256]          ; 32-byte Folded Spill
	stp	q22, q16, [sp, #512]            ; 32-byte Folded Spill
	fmla.8h	v25, v7, v0[0]
	mov.16b	v7, v23
	mov.16b	v0, v12
	subs	x23, x23, #1
	b.eq	LBB1_18
LBB1_22:                                ;   Parent Loop BB1_2 Depth=1
                                        ;     Parent Loop BB1_17 Depth=2
                                        ;       Parent Loop BB1_19 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	stp	q7, q1, [sp, #400]              ; 32-byte Folded Spill
	stp	q5, q4, [sp, #432]              ; 32-byte Folded Spill
	stp	q3, q2, [sp, #464]              ; 32-byte Folded Spill
	str	q6, [sp, #496]                  ; 16-byte Folded Spill
	stp	q25, q24, [x29, #-224]          ; 32-byte Folded Spill
	add	x8, x21, x20
	prfm	pldl1keep, [x8, #16384]
	movi.2d	v1, #0000000000000000
	ldp	q2, q3, [x8]
	ldp	q4, q5, [sp, #352]              ; 32-byte Folded Reload
	fmla.8h	v1, v2, v5
	movi.2d	v2, #0000000000000000
	fmla.8h	v2, v3, v4
	movi.2d	v3, #0000000000000000
	ldp	q4, q5, [x8, #32]
	ldp	q6, q7, [sp, #320]              ; 32-byte Folded Reload
	fmla.8h	v3, v4, v7
	movi.2d	v4, #0000000000000000
	fmla.8h	v4, v5, v6
	prfm	pldl1keep, [x8, #16448]
	ldp	q5, q6, [x8, #64]
	ldr	q7, [sp, #304]                  ; 16-byte Folded Reload
	fmla.8h	v1, v5, v7
	ldp	q7, q5, [sp, #272]              ; 32-byte Folded Reload
	fmla.8h	v2, v6, v5
	ldp	q5, q6, [x8, #96]
	fmla.8h	v3, v5, v7
	ldp	q7, q5, [sp, #240]              ; 32-byte Folded Reload
	fmla.8h	v4, v6, v5
	prfm	pldl1keep, [x8, #16512]
	ldp	q5, q6, [x8, #128]
	fmla.8h	v1, v5, v7
	ldp	q7, q5, [sp, #208]              ; 32-byte Folded Reload
	fmla.8h	v2, v6, v5
	ldp	q5, q6, [x8, #160]
	fmla.8h	v3, v5, v7
	ldp	q7, q5, [sp, #176]              ; 32-byte Folded Reload
	fmla.8h	v4, v6, v5
	prfm	pldl1keep, [x8, #16576]
	ldp	q5, q6, [x8, #192]
	fmla.8h	v1, v5, v7
	ldp	q7, q5, [sp, #144]              ; 32-byte Folded Reload
	fmla.8h	v2, v6, v5
	ldp	q5, q6, [x8, #224]
	fmla.8h	v3, v5, v7
	ldr	q5, [sp, #384]                  ; 16-byte Folded Reload
	fmla.8h	v4, v6, v5
	fadd.8h	v1, v1, v2
	fadd.8h	v2, v3, v4
	fadd.8h	v1, v1, v2
	fcvtl	v2.4s, v1.4h
	ext.16b	v1, v1, v1, #8
	fcvtl	v1.4s, v1.4h
	fadd.4s	v1, v2, v1
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	fmov	s2, w28
	fmul	s1, s1, s2
	fcmp	s1, s0
	fcsel	s12, s1, s0, gt
	b.gt	LBB1_20
; %bb.23:                               ;   in Loop: Header=BB1_22 Depth=4
	fsub	s0, s1, s0
	stp	q18, q17, [x29, #-256]          ; 32-byte Folded Spill
	stp	q20, q19, [sp, #512]            ; 32-byte Folded Spill
	stp	q16, q21, [sp, #480]            ; 32-byte Folded Spill
	stp	q23, q22, [sp, #448]            ; 32-byte Folded Spill
	stp	q27, q26, [sp, #416]            ; 32-byte Folded Spill
	str	q28, [sp, #400]                 ; 16-byte Folded Spill
	stp	q30, q29, [sp, #112]            ; 32-byte Folded Spill
	str	q31, [sp, #96]                  ; 16-byte Folded Spill
	bl	_expf
	ldp	q31, q30, [sp, #96]             ; 32-byte Folded Reload
	ldr	q29, [sp, #128]                 ; 16-byte Folded Reload
	ldp	q28, q27, [sp, #400]            ; 32-byte Folded Reload
	ldp	q26, q23, [sp, #432]            ; 32-byte Folded Reload
	ldp	q22, q16, [sp, #464]            ; 32-byte Folded Reload
	ldp	q21, q20, [sp, #496]            ; 32-byte Folded Reload
	ldr	q19, [sp, #528]                 ; 16-byte Folded Reload
	ldp	q18, q17, [x29, #-256]          ; 32-byte Folded Reload
	fadd	s11, s11, s0
	fcvt	h0, s0
	ldp	q25, q24, [x29, #-224]          ; 32-byte Folded Reload
	b	LBB1_21
LBB1_24:
	mov	w0, #0                          ; =0x0
	add	sp, sp, #656
	ldp	x29, x30, [sp, #144]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #128]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #112]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #96]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #80]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #160            ; 16-byte Folded Reload
	ret
	.loh AdrpAdd	Lloh0, Lloh1
	.loh AdrpAdd	Lloh2, Lloh3
	.cfi_endproc
                                        ; -- End function
	.globl	_dummy_result                   ; @dummy_result
.zerofill __DATA,__common,_dummy_result,4,2
	.section	__TEXT,__const
	.p2align	2, 0x0                          ; @__const.main.seq_lens
l___const.main.seq_lens:
	.long	128                             ; 0x80
	.long	256                             ; 0x100
	.long	512                             ; 0x200
	.long	1024                            ; 0x400
	.long	2048                            ; 0x800

	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"seq_len=%4d: %6.2f ms, %6.2f GFLOPS\n"

.subsections_via_symbols
