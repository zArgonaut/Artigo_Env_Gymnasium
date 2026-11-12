# Demonstration corpus

`articles.jsonl` holds 30 short encyclopedia-style articles in Brazilian Portuguese, one JSON
object per line with the keys `doc_id`, `title`, `body` and `license_id`. They let the package
run games, tests and demos right after installation, without downloading anything.

Every text is original synthetic content written for the ArtigoEnv project and is released under
the MIT licence, like the rest of the repository. The subjects (towns, rivers, animals, plants,
instruments, festivals, dishes, devices and so on) are fictional: nothing here describes real
people, places, organizations or works, and nothing should be read as a factual claim.

The corpus is far too small and too artificial for training or evaluating models; use it only
for smoke tests and demonstrations.
