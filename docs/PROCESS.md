# How the first film was made

This is the story of the first film made with these tools, *Robots Revenge*: a fake
"9 o'clock news" report on a robot invasion. A parent and a kid shot it on an ordinary
camera in an afternoon, so a sequel (or your film) can follow the same path. It took
about two hours to get from raw footage to the film on the family's media server.

Names below are the film's characters, not real people: **Rex Newsome** (the
newsreader, played by the grown-up) and **Penny Sparks** (the roving reporter, played
by the kid, who was also the director).

## 1. Look at the footage before deciding anything

- **Inventory.** Run `ffprobe` on every clip to get its resolution, frame rate and
  length. The camera shot 4K at 25 fps. Duplicate exports (`-2`, `-3`) differed only in
  their headers.
- **Contact sheets.** Make one strip per clip with
  `ffmpeg -vf "fps=4/DUR,scale=480:-1,tile=4x1"`, stack the strips with
  `magick -append`, then *look at them*. That's how we found the plain wall, the robot
  clips and the water attack.
- **Transcripts.** Run `whisper *.wav --model small.en` locally. The transcripts gave us
  the characters and the story, and the running order followed from them.
- **Speech-activity strips.** Print one character per 0.25 s of RMS level (`█` loud,
  `▄` quiet, `·` silent). This finds dead air at the head and tail of a clip, and shows
  where the screams are. It caught a "HELP, HELP!" that a naive trim would have cut in
  half.

## 2. Agree the running order with the kid

The kid sent it a piece at a time while we worked:

> Title card → 9 o'clock news intro → Rex's long intro → Penny live (keyed over the
> water attack, looping) → "help!" keyed over the giant robot walking up → BEEP / error
> / static cut-out → back to Rex, worried → credits → post-credits robot → To Be Continued

Write it down early and treat it as the spec.

## 3. Green screen on a wall that isn't green

The "green" was a sage/olive painted wall, and the light changed between takes, so
iMovie's keyer and ffmpeg's `chromakey` both struggled. `tools/keyer.py` works like
this:

1. **Clean plate.** Take the median of the empty-wall frames at the end of the take.
2. **Relight each frame.** Fit one global colour balance times a smooth local
   *brightness* field, using only the pixels currently believed to be wall. It has to
   be brightness-only. When the field could also shift hue, the kid popping up at the
   lens got learned as wall and keyed out, leaving a hole in their face.
3. **Score pixels mostly on chromaticity.** A shadow on the wall keeps the wall's hue;
   hair and skin don't. Pixels much darker than the plate count as hair.
4. **Clean up the matte.** Keep the largest blobs. Fill holes only where they aren't
   confidently wall, so the gap between a raised arm and the head stays open. Then
   feather, and smooth frame-to-frame only where the matte is steady (otherwise fast
   motion leaves a ghost trail).
5. **Decontaminate the edges and despill.** Solve `frame = a·fg + (1−a)·wall` for fg in
   the soft edge band.

It writes the composite and a "subject on pure green" version (for iMovie's own
Green/Blue Screen tool) in one pass. Long takes go through `tools/render_parallel.sh`,
which runs three slices in parallel. Background plates are looped with
`tools/make_bg_loop.sh`.

**Tip:** film 3–5 s of the empty wall after every green-screen take, in the same light,
and keep the camera still.

## 4. Fan the graphics out to subagents, in parallel

Each graphic is independent, so we gave one agent each piece (four at a time) and had it
build **every option** as a preview. The shared constraints went into every brief.
That's what made the outputs consistent and iMovie-safe:

```text
- 1920x1080, 25 fps, H.264 libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart,
  AAC 48 kHz stereo 192k. Peaks -1 to -3 dBFS.
- Assume ffmpeg has NO drawtext filter. Render text/graphics per frame with
  PIL/numpy/OpenCV and pipe raw frames to ffmpeg (-f rawvideo -pix_fmt rgb24 ...).
  Synthesize all audio with numpy or ffmpeg lavfi. No downloads.
- Fonts: bundled or system fonts only. .ttc needs index=.
  Draw ⚠ yourself; most fonts lack it.
- Scratch in your own folder; don't kill processes you didn't start; nice -n 10 encodes.
- Verify before reporting: extract ~4 frames and LOOK at them; ffprobe size/fps/audio;
  volumedetect. Report paths, durations, one line each on look and sound, caveats.
- Save a re-runnable script in tools/ taking the output path as argv[1].
```

These were the briefs, one line each (a sequel can reuse them with new wording):

| Agent | Brief in one line |
|---|---|
| Titles A+B | "Blockbuster" chrome letters, sparks and BWAAAM; "Comic book" halftone, bouncy letters, KA-POW |
| Titles C+D | "Robot computer" green CRT typing itself out with bleeps; "Warning" hazard stripes, siren, slam |
| News intro | Navy and red broadcast open: globe, clock sweeping to 9:00 with the **six pips**, sting, title, BREAKING ticker; 4 channel-name variants; transparent lower-third PNGs in the same style |
| Signal lost | Freeze on the robot foot's last frame, then digital breakup → static and BZZZ → (A) colour bars and 1 kHz BEEP "We apologise for the loss of signal" / (B) "robots hacked" screen / (C) just static |
| Post-credits | Cut "and action", keep the kid's line, sinister robot voice with `say` and robot FX (check it with whisper), robot's eyes glow red in time with the voice (template-tracked) |
| To Be Continued | Red robot eyes in the dark, chrome-red "To Be Continued", dots landing one at a time, cheeky squint, TV switch-off |

After the agents finished, some things still needed a human or orchestrator pass:
- **Nobody listened to the audio.** The agents balanced it by meter. Check the levels in
  the assembled film, because the titles came out 5–7 dB hotter than the dialogue.
- **Later lead-ins.** When the scene before changes, message the running agent (e.g.
  "start on the robot-walk clip's last frame") rather than restarting it.

## 5. Let the kid choose on a web page

A web page (`tools/picks_page.html`) showed a video for each option and a Pick button.
Picks were saved to the page's database (`choices/family` →
`{title, channel, signal, voice, note}`), and Claude read them back. `tools/picks_page.py`
makes 960 px web copies of whatever previews exist and fills them in, so the page goes
live early and fills up as the agents finish. The kid picked on it while renders ran.
It worked well; reuse it.

## 6. Assemble, check, fix, repeat

`tools/assemble.py --title C --channel A --signal A` conforms every piece to
1080p/25/48k, overlays the lower thirds, normalises loudness, and joins the pieces with
hard cuts. It skips pieces that aren't ready, so the family could watch "the film so
far" early. Each preview turned up real fixes:

- **The best moment was cut off** (the kid popping back up, screaming). We fixed the
  keyer instead of hiding the moment.
- **The wrong water clip.** We swapped the background and re-keyed in parallel slices.
- **The kid was too quiet** (−30 LUFS against the grown-up's −17). `tools/loudness.py`
  compresses, then levels the kid to −15. Plain normalising made it *worse*, because the
  screams set the peak.
- **The titles and the BEEP were too loud.** We normalised the graphics to −17 LUFS in
  assembly.

Measure the loudness of each section of the finished film
(`ffmpeg -ss A -t B -af ebur128` → "I:") before calling it done.

## 7. Deliver

- **iMovie:** use File › Import Media. Import either the finished film, or the pieces
  from the kit folder to keep editing. PNG lower thirds go *above* a clip in the
  timeline.
- **A phone:** encode a ~45 MB copy (`-crf 24 -maxrate 3M`) and send it to yourself.
- **A home media server (Plex, Jellyfin):** give it its own folder, rate it **U**/**G**
  so kids' profiles can see it, lock the title and summary so the server doesn't match
  it to a real film, and upload the poster and fanart from `tools/cover_art.py`.

## Where things live

| What | Where | In git? |
|---|---|---|
| Raw footage | `footage/` beside your project | no |
| Rendered pieces, previews, final film | `kit/` beside your project | no |
| Tools, page, docs | this repo | yes |
