"""测试 SkillRegistry。"""

import pytest
from pathlib import Path
from agents.skill_registry import Skill, SkillRegistry


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """创建临时 skills 目录结构。"""
    # price skill with state_hooks
    price_dir = tmp_path / "price"
    price_dir.mkdir()
    (price_dir / "skill.md").write_text(
        "---\nname: price\ndescription: 处理议价\nstate_hooks:\n  - bargain_count\n  - min_price\n---\n\n议价 prompt {bargain_count} {min_price}"
    )

    # product skill without state_hooks
    product_dir = tmp_path / "product"
    product_dir.mkdir()
    (product_dir / "skill.md").write_text(
        "---\nname: product\ndescription: 处理商品咨询\n---\n\n商品咨询 prompt"
    )

    # invalid skill missing description
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "skill.md").write_text(
        "---\nname: bad\n---\n\n没有 description 的 skill"
    )

    return tmp_path


def test_load_skills_basic(skills_dir: Path):
    """测试基本加载：price 和 product 成功，bad 被跳过。"""
    registry = SkillRegistry(skills_dir)
    skills = registry.list_skills()
    names = {s.name for s in skills}
    assert "price" in names
    assert "product" in names
    assert "bad" not in names


def test_skill_state_hooks(skills_dir: Path):
    """测试 state_hooks 正确解析。"""
    registry = SkillRegistry(skills_dir)
    price = registry.get_skill("price")
    assert price.state_hooks == ["bargain_count", "min_price"]


def test_skill_no_state_hooks(skills_dir: Path):
    """测试没有 state_hooks 时默认为空列表。"""
    registry = SkillRegistry(skills_dir)
    product = registry.get_skill("product")
    assert product.state_hooks == []


def test_skill_prompt_body(skills_dir: Path):
    """测试 prompt 正文（frontmatter 之后的内容）正确提取。"""
    registry = SkillRegistry(skills_dir)
    product = registry.get_skill("product")
    assert "商品咨询 prompt" in product.prompt
    assert "---" not in product.prompt


def test_skill_dir_path(skills_dir: Path):
    """测试 skill_dir 指向正确的目录。"""
    registry = SkillRegistry(skills_dir)
    price = registry.get_skill("price")
    assert price.skill_dir == skills_dir / "price"


def test_get_unknown_skill_returns_none(skills_dir: Path):
    """测试获取不存在的 skill 返回 None。"""
    registry = SkillRegistry(skills_dir)
    result = registry.get_skill("nonexistent")
    assert result is None


def test_build_classify_context(skills_dir: Path):
    """测试生成 classify prompt 注入内容。"""
    registry = SkillRegistry(skills_dir)
    context = registry.build_classify_context()
    assert "price" in context
    assert "处理议价" in context
    assert "product" in context
    assert "处理商品咨询" in context


def test_skills_dir_not_exists():
    """测试 skills 目录不存在时抛出错误。"""
    with pytest.raises(FileNotFoundError, match="skills"):
        SkillRegistry(Path("/nonexistent/path/skills"))
