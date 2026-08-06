import numpy as np
import pytest

from test_pca_artifact import make_result


@pytest.mark.parametrize("mean_sub", [False, True])
def test_all_components_reconstruct_source_cube(mean_sub):
    result = make_result(mean_sub=mean_sub)
    expected = np.arange(12, dtype=float).reshape(3, 2, 2)

    np.testing.assert_allclose(result.reconstruct(), expected)


def test_include_and_exclude_component_selection():
    result = make_result()

    included = result.reconstruct(include=[0, 2])
    expected = result.eigenimages.copy()
    expected[1] = 0.0
    np.testing.assert_allclose(included, expected)

    excluded = result.reconstruct(exclude=[1])
    np.testing.assert_allclose(excluded, expected)


def test_exclude_tail_and_empty_reconstruction():
    result = make_result()

    reconstructed = result.reconstruct(exclude_tail=1)
    expected = np.zeros_like(result.eigenimages)
    expected[0] = result.eigenimages[0]
    np.testing.assert_allclose(reconstructed, expected)

    np.testing.assert_allclose(
        result.reconstruct(exclude_tail=0),
        np.zeros_like(result.eigenimages),
    )


def test_invalid_component_is_rejected():
    result = make_result()

    with pytest.raises(IndexError, match="available range"):
        result.reconstruct(exclude=[result.n_components])
