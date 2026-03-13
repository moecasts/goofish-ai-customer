# Project Guideline

## Git Commit Message Convention

This project follows **Conventional Commits** specification for clear, semantic commit messages.

### Format

```
type(scope): description
```

### Components

#### Type (required)

Describes the kind of change:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes only
- **style**: Code style changes (formatting, missing semicolons, etc.) - no logic changes
- **refactor**: Code refactoring without feature or bug fix
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Build process, dependencies, tooling, configuration - no production code changes
- **ci**: CI/CD configuration changes
- **revert**: Revert a previous commit

#### Scope (optional)

Specifies what part of the codebase is affected:

- `agent`: XianyuAgent.py (main agent logic)
- `api`: XianyuApis.py (API integration)
- `context`: context_manager.py (context management)
- `utils`: Utilities in utils/ directory
- `ws`: WebSocket client in main.py
- `config`: Configuration files
- `docker`: Docker and containerization
- `tests`: Test files

#### Description (required)

Clear, concise explanation of what changed:

- Use imperative mood ("add" not "added" or "adds")
- Don't capitalize first letter (unless scope is included)
- No period (.) at the end
- Keep under 50 characters when possible

### Examples

#### Good Commits ✅

```
feat(agent): add multi-expert routing system
fix(api): handle token refresh timeout correctly
docs: update API integration guide
refactor(context): simplify SQLite query logic
perf(ws): optimize message parsing performance
test(agent): add unit tests for intent router
chore: update dependencies in requirements.txt
ci: configure GitHub Actions workflow
```

#### Bad Commits ❌

```
Fixed bug                              ## Too vague, missing type/scope
feat: Fixed the agent routing bug      ## Wrong tense, not imperative
FEAT(AGENT): Add Feature               ## Wrong case, too generic
type(scope): description.              ## Period at end
fix(agent) update something            ## Missing colon
```

### Tips

- **Atomic commits**: Each commit should represent one logical change
- **Meaningful scope**: Use scope to clarify what area changed
- **Reviewability**: Your future self and teammates will appreciate clear messages
- **Searchability**: Good commits are easier to find and understand in history

### Reference

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
