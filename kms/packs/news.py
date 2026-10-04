"""The news pack: a TV news report. Titles, the 9 o'clock open with the pips, name captions,
a signal-lost cut-out, the villain's voice, credits and an end card."""
import hashlib
import re
from pathlib import Path

from kms import media
from kms.packs import Group, Option, Pack

HOURS = {"ONE": 13, "TWO": 14, "THREE": 15, "FOUR": 16, "FIVE": 17, "SIX": 18, "SEVEN": 19, "EIGHT": 20,
         "NINE": 21, "TEN": 22, "ELEVEN": 23, "TWELVE": 12}

DIRECTOR_TITLES = [
    ("director", "Director", "The classic."),
    ("big-boss", "Big Boss Director", "Everyone knows who's in charge."),
    ("supreme", "Supreme Movie Commander", "For directors who mean business."),
    ("chief", "Chief of Absolutely Everything", "Because it's true."),
]


def hour_of(programme):
    """21 for "THE NINE O'CLOCK NEWS", 18 for "6 O'CLOCK NEWS"; 21 when it doesn't say."""
    up = programme.upper()
    m = re.search(r"\b(\d{1,2})\s*O[’']?CLOCK", up)
    if m:
        h = int(m.group(1)) % 24
        return h + 12 if 1 <= h <= 11 else h
    for word, h in HOURS.items():
        if re.search(rf"\b{word}\s+O[’']?CLOCK", up):
            return h
    return 21


class NewsPack(Pack):
    name = "news"

    # ------------------------------------------------------------------ choices
    def groups(self, p):
        villain = p.story_text("villain", "ROBOT").strip() or "ROBOT"
        v = villain.capitalize()
        return [
            Group("title", "Opening title", "The very first thing on screen.", [
                Option("blockbuster", "Blockbuster", "Chrome letters, sparks and a deep cinema BWAAAM."),
                Option("comic", "Comic book", "Bright halftone dots, bouncy letters and a KA-POW!"),
                Option("computer", f"{v} computer", "A green computer screen typing itself out, with beeps and boops."),
                Option("warning", "Warning siren", "Flashing hazard stripes, a siren, then the title slams in."),
            ], default="computer"),
            Group("channel", "News channel", "Shown in the news intro, with the clock and the pips.",
                  self._channel_options(p)),
            Group("signal", "Signal lost", f"What happens when the {villain.lower()} smashes the camera.", [
                Option("glitch-beep", "Glitch, static and BEEEP",
                       "The picture breaks up, BZZZZ static, then colour bars and “We apologise for the loss of signal”."),
                Option("hacked", f"{v}s hacked it", f"Glitch and static, then a red ERROR screen: the {villain.lower()}s take over."),
                Option("static", "Just static", "Short and punchy: glitch, two seconds of BZZZZ, straight back to the studio."),
            ]),
            Group("voice", "The villain's voice", f"“{p.story_text('villain_line', '')}” Turn the sound up!", [
                Option(tid, name, desc) for tid, (name, desc, _) in _voice_takes().items()
            ]),
            *self._eyes_group(p),
            Group("ending", "The very end", "The last thing on screen.", [
                Option("to-be-continued", p.story_text("ending", "To Be Continued") or "To Be Continued",
                       "Glowing eyes in the dark, the words, and the dots landing one by one. There'll be a sequel!"),
                Option("the-end", "The End", "Glowing eyes in the dark, then THE END. Finished!"),
            ]),
            Group("director_title", "Your name in the credits",
                  f"How should it say {p.director}'s job at the end?", [
                      Option(i, name, desc) for i, name, desc in DIRECTOR_TITLES], kind="text"),
            Group("cover", "Poster picture", "The picture on the film's poster.", self._cover_options(p), kind="image"),
        ]

    def _villain_scene(self, p):
        return next((s for s in p.scenes if s.kind == "make" and s.target == "villain-line"), None)

    def _eyes_group(self, p):
        s = self._villain_scene(p)
        if not (s and s.get("eyes")):
            return []
        return [Group("eyes", "Glowing eyes", "Should the villain's eyes glow when it talks?", [
            Option("glow", "Red glowing eyes", "The eyes light up red in time with the voice. Spooky!"),
            Option("none", "No glow", "Just the voice. Sneaky!"),
        ], default="glow")]

    def voice_line(self, p):
        """(line, name) for the speech engine: `hero_say` stands in for a name it mispronounces."""
        line, hero = p.story_text("villain_line", "We'll see..."), p.story_text("hero", "")
        say = p.story_text("hero_say", "")
        if hero and say:
            line = re.sub(re.escape(hero), say, line, flags=re.I)
            hero = say
        return line, hero

    def voice_wav(self, p, take):
        """The picked take of the villain's line, made once and kept (kit/pieces/voice-*.wav)."""
        from kms.render import voices as m
        line, hero = self.voice_line(p)
        tag = hashlib.sha1(f"{take}|{line}|{hero}".encode()).hexdigest()[:8]
        wav = p.kit_dir("pieces") / f"voice-{take}-{tag}.wav"
        if not wav.exists():
            m.render_voice(wav, take=take, line=line, hero=hero)
        return wav

    def _channel_options(self, p):
        opts = []
        for i, ch in enumerate(p.data.get("channels") or [{"name": "{initial}BC NEWS"}], start=1):
            ch = ch if isinstance(ch, dict) else {"name": str(ch)}
            name = p.fill(ch.get("name", "")).strip()
            strap = p.fill(ch.get("strapline", "")).strip()
            label = name or p.story_text("programme", "THE NINE O'CLOCK NEWS")
            desc = strap or ("No channel name, just the programme title." if not name else "")
            opts.append(Option(f"ch{i}", label, desc, {"channel": name, "strapline": strap}))
        return opts

    def _cover_options(self, p):
        cands = [c for c in (p.kit / "previews" / "cover").glob("frame-*.jpg") if not c.stem.endswith("-poster")]
        cands.sort(key=lambda c: int(c.stem.split("-")[1]))
        return [Option(c.stem, f"Picture {i}", "") for i, c in enumerate(cands, start=1)]

    # ------------------------------------------------------------------ rendering
    def preview_kind(self, group):
        return {"cover": ".jpg"}.get(group, ".mp4")

    def render_option(self, p, group, option, out, ctx):
        """Render one option as a full-quality piece (previews ARE the pieces)."""
        g = self.group(p, group)
        opt = g.option(option) if g else None
        if opt is None:
            raise KeyError(f"no option {option!r} in {group!r}")
        title, villain = p.title, p.story_text("villain", "ROBOT") or "ROBOT"
        programme = p.story_text("programme", "THE NINE O'CLOCK NEWS")
        if group == "title":
            if option == "blockbuster":
                from kms.render import title_blockbuster as m
                return m.render(out, title=title, ctx=ctx)
            if option == "comic":
                from kms.render import title_comic as m
                return m.render(out, title=title, ctx=ctx)
            if option == "computer":
                from kms.render import title_computer as m
                return m.render(out, title=title, villain=villain, subtitle=f"> LIVE: {programme}", ctx=ctx)
            if option == "warning":
                from kms.render import title_warning as m
                return m.render(out, title=title, villain=villain, subtitle=f"TONIGHT ON {programme}", ctx=ctx)
        if group == "channel":
            from kms.render import news_intro as m
            return m.render(out, programme=programme, channel=opt.params["channel"],
                            strapline=opt.params["strapline"] or None, crawl=p.story_text("crawl", None) or None,
                            headline=p.story_text("headline", "") or None, hour=hour_of(programme), ctx=ctx)
        if group == "signal":
            from kms.render import signal_lost as m
            source, start = self.lead_in(p)
            return m.render(out, style=option, source=source, source_start=start,
                            ident=f"{title} NEWS", villain=villain, ctx=ctx)
        if group == "voice":
            from kms.render import voices as m
            clip, at = self._villain_picture(p)
            line, hero = self.voice_line(p)
            return m.render(out, take=option, line=line, hero=hero, picture=clip, picture_at=at, ctx=ctx)
        if group == "eyes":
            s = self._villain_scene(p)
            clip = p.footage_file(s.get("clip")) if s.get("clip") else None
            if clip is None or not media.readable(clip):  # not filmed yet: show it on a stand-in robot
                from kms import demo
                clip = p.kit_dir("previews", "eyes") / "stand-in-robot.mp4"
                if not media.readable(clip):
                    demo.robot_chair(clip, fps=p.fps, seconds=7.0)
                s = type(s)(s.id, s.kind, s.target, {**s.raw, "from": 0, "to": None})
            take, _ = self.choice(p, "voice")
            return self._speak_over(p, s, clip, self.voice_wav(p, take), out, ctx, glow=option == "glow")
        if group == "ending":
            from kms.render import end_card as m
            if option == "the-end":
                return m.render(out, text="The End", dots=False, ctx=ctx)
            return m.render(out, text=p.story_text("ending", "To Be Continued") or "To Be Continued",
                            dots=True, ctx=ctx)
        if group == "cover":
            from kms.render import cover_art as m
            frame = Path(out).parent / f"{option}.jpg"
            return m.poster(frame, Path(out).with_name(f"{option}-poster.jpg"), **self.cover_words(p))
        raise KeyError(f"the news pack can't render {group}/{option}")

    def cover_words(self, p):
        programme = p.story_text("programme", "THE NINE O'CLOCK NEWS")
        h = hour_of(programme) % 12 or 12
        starring = " & ".join(dict.fromkeys(c.get("actor", "") for c in p.cast if c.get("actor"))).upper()
        channel_id, _ = self.choice(p, "channel")
        ch = self.group(p, "channel").option(channel_id) if channel_id else None
        presenter = (ch.params["channel"] if ch and ch.params["channel"] else "").upper()
        credit = " · ".join(x for x in (f"{presenter} PRESENTS" if presenter else "",
                                        f"STARRING {starring}" if starring else "") if x)
        kicker = tuple(x for x in (f"TONIGHT AT {h} O'CLOCK…", (p.tagline or "").upper()) if x)
        return {"title": p.title, "kicker": kicker, "credit": credit}

    def lead_in(self, p):
        """The clip and start time the signal-lost piece breaks up from, if it's been filmed."""
        sig = next((s for s in p.scenes if s.kind == "pick" and s.target == "signal"), None)
        ref = p.scene(sig.get("lead_in")) if sig and sig.get("lead_in") else None
        if ref is None or ref.kind not in ("clip", "key"):
            return None, None
        f = (p.kit / "keyed" / f"{ref.id}.mp4") if ref.kind == "key" else p.footage_file(ref.target)
        if not media.readable(f):
            return None, None
        end = float(ref.get("to") or media.duration(f))
        return str(f), max(0.0, end - 0.24)

    def _villain_picture(self, p):
        s = next((s for s in p.scenes if s.kind == "make" and s.target == "villain-line"), None)
        if s and s.get("clip") and media.readable(p.footage_file(s.get("clip"))):
            return str(p.footage_file(s.get("clip"))), float(s.get("from") or 0)
        return None, 0.0

    # ------------------------------------------------------------------ made from movie.yaml
    def credits_entries(self, p):
        title_id, _ = self.choice(p, "director_title")
        job = dict((i, n) for i, n, _ in DIRECTOR_TITLES).get(title_id, "Director")
        e = [("title", p.title), ("gap", 120), ("role", job, p.director)]
        if p.grownup:
            e += [("role", "Second-in-Command Director", p.grownup), ("role", "Tech Editor", p.grownup),
                  ("role", "Producers", f"{p.grownup} & {p.director}")]
        if p.cast:
            e += [("gap", 110), ("heading", "CAST"), ("gap", 30)]
            e += [("role", c["character"], c.get("actor", "")) for c in p.cast]
        villain = p.story_text("villain", "ROBOT").title()
        e += [("gap", 160), ("small", f"Filmed on location at {villain} HQ"),
              ("gap", 40), ("small", "Made with kid-movie-studio")]
        return e

    def make(self, p, scene, out, ctx):
        if scene.target == "credits":
            from kms.render import credits as m
            return m.render(out, title=p.title, entries=self.credits_entries(p), ctx=ctx)
        if scene.target == "villain-line":
            return self._villain_line(p, scene, out, ctx)
        raise KeyError(f"the news pack can't make {scene.target!r}")

    def _villain_line(self, p, scene, out, ctx):
        """The villain's voice (the kid's pick) over the close-up, eyes glowing if they picked that."""
        take, _ = self.choice(p, "voice")
        clip = p.footage_file(scene.get("clip"))
        if not media.readable(clip):
            raise FileNotFoundError(f"{clip.name} isn't in footage/ yet")
        eyes, _ = self.choice(p, "eyes") if scene.get("eyes") else (None, False)
        return self._speak_over(p, scene, clip, self.voice_wav(p, take), out, ctx, glow=eyes == "glow")

    def _speak_over(self, p, scene, clip, wav, out, ctx, glow=False):
        at = float(scene.get("voice_at") or 1.0)
        start = float(scene.get("from") or 0)
        end = scene.get("to")
        if glow:
            from kms.render import glowing_eyes as m
            return m.render(out, clip=clip, voice_wav=wav, voice_at=at, start=start,
                            end=float(end) if end is not None else None, ctx=ctx)
        vdur = media.duration(wav) or 4
        length = float(end) - start if end is not None else min((media.duration(clip) or 0) - start, at + vdur + 1.0)
        if ctx.limit_frames:
            length = min(length, ctx.limit_frames / ctx.fps)
        ms = int(at * 1000)
        media.run([media.ffmpeg(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", clip,
                   "-i", wav, "-filter_complex",
                   f"[0:v]scale=1920:1080:flags=lanczos,fps={ctx.fps},format=yuv420p,setsar=1[v];"
                   f"[0:a]aresample=48000,aformat=channel_layouts=stereo,"
                   f"volume='if(between(t,{at},{at + vdur}),0.35,1)':eval=frame[bed];"
                   f"[1:a]adelay={ms}|{ms}[vo];[bed][vo]amix=inputs=2:duration=first:normalize=0[a]",
                   "-map", "[v]", "-map", "[a]", *media.video_args(18), *media.audio_args(),
                   "-movflags", "+faststart", out])
        return Path(out)

    def captions(self, p, out_dir):
        """Name captions (lower thirds) for every scene with `caption:`; {scene id: png}."""
        from kms.render import news_intro as m
        roles = {c["character"]: c.get("role", "") for c in p.cast}
        out = {}
        for s in p.scenes:
            name = s.get("caption")
            if not name:
                continue
            png = Path(out_dir) / f"{s.id}.png"
            m.lower_third(png, name=str(name).upper(), role=roles.get(name, ""), live=bool(s.get("caption_live")))
            out[s.id] = png
        return out


def _voice_takes():
    from kms.render.voices import TAKES
    return TAKES
