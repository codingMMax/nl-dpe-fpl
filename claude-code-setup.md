# Claude Code Environment Setup

Paste this file into a new Claude Code session and ask: **"Install all the plugins and skills listed here."**

---

## Plugins to Install

### 1. Superpowers (anthropics/claude-plugins-official)

The core skill framework. Provides 15+ skills for structured workflows: brainstorming, TDD, debugging, planning, code review, parallel agents, git worktrees, and more.

- **Source:** `anthropics/claude-plugins-official`
- **Plugin name:** `superpowers`

### 2. Claude HUD (jarrodwatts/claude-hud)

Real-time statusline showing model info, token usage, tool activity, git state, and todo progress.

- **Source:** `jarrodwatts/claude-hud`
- **Plugin name:** `claude-hud`

### 3. Planning with Files (OthmanAdi/planning-with-files)

Manus-style file-based planning. Creates `task_plan.md`, `findings.md`, `progress.md` for complex multi-step tasks. Supports session recovery after `/clear`.

- **Source:** `OthmanAdi/planning-with-files`
- **Plugin name:** `planning-with-files`

---

## Settings to Apply

After installing plugins, apply these global settings in `~/.claude/settings.json`:

```json
{
  "effortLevel": "high",
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "claude-hud@claude-hud": true,
    "planning-with-files@planning-with-files": true
  },
  "extraKnownMarketplaces": {
    "claude-plugins-official": {
      "source": { "source": "github", "repo": "anthropics/claude-plugins-official" }
    },
    "claude-hud": {
      "source": { "source": "github", "repo": "jarrodwatts/claude-hud" }
    },
    "planning-with-files": {
      "source": { "source": "github", "repo": "OthmanAdi/planning-with-files" }
    }
  }
}
```

## Post-Install

1. Run `/claude-hud:setup` to configure the statusline
2. Verify skills load by checking that `superpowers:brainstorming` and other skills appear in the session
