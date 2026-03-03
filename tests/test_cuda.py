import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA를 사용할 수 없는 환경입니다.",
)


@pytest.mark.unit
def test_cuda_is_available():
    """torch.cuda.is_available()이 True를 반환한다."""
    assert torch.cuda.is_available()


@pytest.mark.unit
def test_cuda_device_count():
    """CUDA 디바이스가 1개 이상 존재한다."""
    assert torch.cuda.device_count() >= 1


@pytest.mark.unit
def test_cuda_version():
    """CUDA 버전이 12 이상이다."""
    assert torch.version.cuda is not None
    major = int(torch.version.cuda.split(".")[0])
    assert major >= 12


@pytest.mark.unit
def test_gpu_name():
    """GPU 이름 문자열이 반환된다."""
    name = torch.cuda.get_device_name(0)
    assert isinstance(name, str)
    assert len(name) > 0


@pytest.mark.unit
def test_tensor_on_cuda():
    """텐서를 CUDA 디바이스로 이동할 수 있다."""
    t = torch.tensor([1.0, 2.0, 3.0]).cuda()
    assert t.device.type == "cuda"


@pytest.mark.unit
def test_tensor_arithmetic_on_cuda():
    """CUDA 텐서 간 덧셈 결과가 정확하다."""
    a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
    assert torch.equal(a + b, torch.tensor([5.0, 7.0, 9.0], device="cuda"))


@pytest.mark.unit
def test_tensor_matmul_on_cuda():
    """CUDA에서 행렬 곱셈 shape과 값이 정확하다."""
    a = torch.ones(3, 4, device="cuda")
    b = torch.ones(4, 5, device="cuda")
    result = a @ b
    assert result.shape == (3, 5)
    assert torch.all(result == 4.0)


@pytest.mark.unit
def test_float16_tensor_on_cuda():
    """float16(half precision) 텐서가 CUDA에서 생성된다."""
    t = torch.zeros(2, 2, dtype=torch.float16, device="cuda")
    assert t.dtype == torch.float16
    assert t.device.type == "cuda"


@pytest.mark.unit
def test_bfloat16_tensor_on_cuda():
    """bfloat16 텐서가 CUDA에서 생성된다."""
    t = torch.zeros(2, 2, dtype=torch.bfloat16, device="cuda")
    assert t.dtype == torch.bfloat16
    assert t.device.type == "cuda"


@pytest.mark.unit
def test_gpu_memory_total():
    """GPU 전체 메모리가 1GB 이상이다."""
    total = torch.cuda.get_device_properties(0).total_memory
    assert total >= 1 * 1024 ** 3


@pytest.mark.unit
def test_cuda_synchronize():
    """대규모 행렬 곱 후 cuda.synchronize()가 예외 없이 완료된다."""
    t = torch.randn(1000, 1000, device="cuda")
    _ = t @ t
    torch.cuda.synchronize()
