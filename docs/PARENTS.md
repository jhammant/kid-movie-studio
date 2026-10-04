# A parent's guide to filming day

You and your kid are a team. You run the camera and the computer; they're the director, so
they decide what the film looks like. This guide is for you. `kms next` gives you the same
advice one step at a time.

## Before you start

- **A plain wall.** Any colour, as long as it's plain: no pictures, shelves or switches in
  the shot. Light it evenly (daylight from a window to the side is great), and stand the
  actors a step or two in front of it so they don't throw hard shadows on it.
- **Something to hold the camera still.** A tripod, or a pile of books. A phone works fine.
  Film sideways (landscape).
- **Clothes that aren't the wall's colour.** The studio removes the wall colour, so
  anything that matches it goes see-through.
- **Toys, costumes, a hose, a cardboard microphone.** The props are half the fun.
- **About two hours, in sittings.** Most kids have about 20 minutes of focus in one go.
  Plan in the morning, film after lunch, premiere at teatime.

## How to keep the kid in charge

The studio makes every creative decision a choice between finished previews. Your job is
to protect that:

- **Ask, don't tell.** "What's our film called?" "Who's the villain?" "Which one makes you
  laugh?"
- **Don't vote.** If they pick the one you like least, that's the one. You can suggest;
  they decide. (Grown-ups can lock a choice in `movie.yaml` if they really must, say
  because something is too scary for a little sibling. Use it sparingly.)
- **Let them say "Action!" and "Cut!"** It's the director's job, and it means they decide
  when a take is good.
- **Type their ideas into the "Anything to change?" box.** Then make new options from them
  (`kms previews --fresh title`, or new words in `movie.yaml`) and let them pick again.
- **Credit them.** The credits list them as the Director by default, and they choose
  their own job title.

## Filming the plain-wall shots

1. Camera still, wall lit, everyone in place.
2. "Action!" Film the scene.
3. **Everyone steps out, and you keep filming the empty wall for 5 seconds.** Same light,
   don't touch the camera. This "clean plate" is what makes the magic work.
4. "Cut!"

If you forget step 3, film the empty wall straight afterwards, without moving the camera,
and set `plate: {clip: that-file.mp4, from: 0, for: 4}` for the scene.

## When things go wrong

- **An arm or a hat goes see-through:** the clothes matched the wall. Film that bit again
  in something different, or keep it: in a robot attack, nobody minds.
- **The kid is too quiet next to the grown-up:** the studio levels everyone to the same
  loudness. To bring the kid forward, give their scenes `level: -15` in `movie.yaml`.
- **They want to change everything after the premiere:** brilliant. `kms picks`, then
  `kms assemble`. Each round takes minutes, not hours.
- **Attention runs out:** stop. Everything is saved, and `kms next` picks up where you
  left off.

## Privacy and sharing

- Everything stays on your computer: footage, previews and the finished film. The pick
  page is served from your computer to your own Wi-Fi, behind a key in its link.
- `movie.yaml` holds first names or nicknames only. You don't need surnames.
- Sharing is your decision. The film has your family's faces and voices in it. Send it to
  grandparents, sure; think twice about anywhere public.
