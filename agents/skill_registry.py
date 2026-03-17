"""Skills 注册表：扫描并管理 config/skills/ 目录下的所有 skill。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
from loguru import logger


@dataclass
class Skill:
    """单个 skill 的数据结构。"""

    name: str
    description: str
    prompt: str
    state_hooks: list[str] = field(default_factory=list)
    write_hooks: list[str] = field(default_factory=list)
    skill_dir: Optional[Path] = None


class SkillRegistry:
    """扫描 skills 目录，管理所有已注册 skill。"""

    def __init__(self, skills_dir: Path):
        if not skills_dir.exists():
            raise FileNotFoundError(f"skills 目录不存在: {skills_dir}")
        self._skills: dict[str, Skill] = {}
        self._load_skills(skills_dir)

    def _load_skills(self, skills_dir: Path) -> None:
        """扫描 skills_dir/*/skill.md 并加载所有有效 skill。"""
        for skill_md in sorted(skills_dir.glob("*/skill.md")):
            skill = self._parse_skill(skill_md)
            if skill:
                self._skills[skill.name] = skill
                logger.info(f"已加载 skill: {skill.name}")

    def _parse_skill(self, skill_md: Path) -> Optional[Skill]:
        """解析单个 skill.md 文件，返回 Skill 或 None（解析失败时）。"""
        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, prompt = self._split_frontmatter(content)
            if frontmatter is None:
                logger.warning(f"skill 缺少 frontmatter: {skill_md}")
                return None

            meta = yaml.safe_load(frontmatter)
            if not meta.get("name") or not meta.get("description"):
                logger.warning(f"skill 缺少 name 或 description，跳过: {skill_md}")
                return None

            return Skill(
                name=meta["name"],
                description=meta["description"],
                prompt=prompt.strip(),
                state_hooks=meta.get("state_hooks") or [],
                write_hooks=meta.get("write_hooks") or [],
                skill_dir=skill_md.parent,
            )
        except Exception as e:
            logger.warning(f"解析 skill 失败，跳过 {skill_md}: {e}")
            return None

    def _split_frontmatter(self, content: str) -> tuple[Optional[str], str]:
        """分割 YAML frontmatter 和 Markdown 正文。"""
        if not content.startswith("---"):
            return None, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None, content
        return parts[1], parts[2]

    def get_skill(self, name: str) -> Optional[Skill]:
        """按 name 获取 skill，不存在时返回 None。"""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """返回所有已注册 skill 列表。"""
        return list(self._skills.values())

    def build_classify_context(self) -> str:
        """生成注入 classify prompt 的 skill 列表描述。

        技能按目录名称字母顺序排列（由 _load_skills 中的 sorted() 保证）。
        """
        lines = [f"- {s.name}: {s.description}" for s in self._skills.values()]
        return "\n".join(lines)
