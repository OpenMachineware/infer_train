# core/utils.mojo
#
# Small shared helpers used across the runtime.  Kept separate so that the
# more "architectural" modules stay focused on their own concern.
#
# Mojo 1.0 note: there is no built-in `unimplemented()`; we model it with the
# `abort()` intrinsic from `std.os.os`, which terminates via a target
# dependent trap instruction.

from std.os.os import abort


def unimplemented(message: String = "not implemented") -> None:
    """Terminate on a not-yet-implemented code path.

    Used as the body of `backward` stubs and other M1 placeholders.  The
    message is intentionally unused: Mojo 1.0 has no lightweight way to print
    and then trap without pulling in `Writable` formatting, so we keep this
    dependency-free.
    """
    _ = message
    abort()


def align_up(value: Int, alignment: Int) -> Int:
    """Round `value` up to the nearest multiple of `alignment`."""
    return (value + alignment - 1) // alignment * alignment


def div_ceil(numerator: Int, denominator: Int) -> Int:
    """Integer division rounding up."""
    return (numerator + denominator - 1) // denominator


def make_shape2(d0: Int, d1: Int) -> List[Int]:
    """Build a rank-2 shape list [d0, d1]."""
    var shape = List[Int]()
    shape.append(d0)
    shape.append(d1)
    return shape^


def make_shape1(d0: Int) -> List[Int]:
    """Build a rank-1 shape list [d0]."""
    var shape = List[Int]()
    shape.append(d0)
    return shape^
