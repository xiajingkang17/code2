from plan.narration import build_narration
from plan.schema import Step


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_fraction_and_power() -> None:
    step = Step(line="$\\frac{1}{2}x^2$", subtitle="对 $\\frac{1}{2}x^2$ 求导")
    speech = _norm(build_narration(step))
    assert "1 分之 2" in speech
    assert "x 的 2 次方" in speech


def test_subscript_and_equals() -> None:
    step = Step(line="$a_1=2$", subtitle="已知 $a_1=2$")
    speech = _norm(build_narration(step))
    assert "a 下标 1" in speech
    assert "等于 2" in speech


def test_sqrt_times() -> None:
    step = Step(line="$\\sqrt{3}\\times x$", subtitle="计算 $\\sqrt{3}\\times x$")
    speech = _norm(build_narration(step))
    assert "根号 3" in speech
    assert "乘" in speech


def test_sum_limits() -> None:
    step = Step(line="$\\sum_{i=1}^{n} i$", subtitle="求 $\\sum_{i=1}^{n} i$")
    speech = _norm(build_narration(step))
    assert "从 1 到 n 求和" in speech


def test_derivative_fraction() -> None:
    step = Step(line="$\\frac{d}{dx}x^2$", subtitle="对 $x^2$ 求导")
    speech = _norm(build_narration(step))
    assert "对 x 求导" in speech
