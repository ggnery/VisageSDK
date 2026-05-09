"""Tests for the registry helper."""

import pytest

from registry import Registry


class TestRegistry:
    def test_register_and_get(self):
        r = Registry("widget")
        class Foo: ...
        r.register("foo", Foo)
        assert r.get("foo") is Foo

    def test_register_duplicate_raises(self):
        r = Registry("widget")
        class A: ...
        class B: ...
        r.register("x", A)
        with pytest.raises(ValueError, match="already registered"):
            r.register("x", B)

    def test_get_unknown_raises_with_available(self):
        r = Registry("widget")
        r.register("foo", object)
        r.register("bar", object)
        with pytest.raises(KeyError) as exc:
            r.get("baz")
        msg = str(exc.value)
        assert "widget 'baz' not found" in msg
        assert "bar, foo" in msg  # sorted listing

    def test_names_returns_registered(self):
        r = Registry("kind")
        r.register("a", object)
        r.register("b", object)
        assert sorted(r.names()) == ["a", "b"]

    def test_get_unknown_with_empty_registry(self):
        r = Registry("kind")
        with pytest.raises(KeyError, match="<none>"):
            r.get("anything")


class TestGlobalRegistries:
    """Verify the side-effect imports populate every registry as expected."""

    def test_all_registries_populated(self, populated_registries):
        b = populated_registries
        assert "inception_resnet_v1" in b.BACKBONES.names()
        assert "mobilenetv3" in b.BACKBONES.names()
        assert "triplet" in b.LOSSES.names()
        assert "cross_entropy" in b.LOSSES.names()
        assert "image_folder" in b.DATASETS.names()
        assert "lfw_pairs" in b.EVAL_DATASETS.names()
        assert "identification" in b.EVAL_DATASETS.names()
        assert "facenet" in b.SAMPLERS.names()
        assert "adaptative" in b.EARLY_STOPPERS.names()
        assert "verification" in b.EVALUATORS.names()
        assert "identification" in b.EVALUATORS.names()
        # Single TRANSFORMATIONS registry with prefixed names
        for n in ("vgg_face2_train", "vgg_face2_val",
                  "casia_webface_train", "casia_webface_val", "lfw_eval"):
            assert n in b.TRANSFORMATIONS.names()
