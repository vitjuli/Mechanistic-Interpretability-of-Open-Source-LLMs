# Decode under steering — electron/proton pair, L24, c=16 (push norm 100.2)

Source: CSD3 run `logs/decode-31165192.out` (Sat 27 Jun 2026). Model Qwen3-4B base.
Steering at the answer position along delta (Δμ), usage (u) vs baseline; prompts correct=electron, incorrect=proton.
Evidence for the qualitative dissociation (Section 3.3): delta = coherent content (verbose, top token = function word), usage = forced token "Photon" (intact≈1) but degenerate repetition.

## delta (Δμ) — coherent physics content about the photon, top token = "The"/"A"/"Phot" (intact low)
- p0: electron rank 51 -> 4049. gen: "Photons of ultraserentate are the carriers of the energy in the quanta of the electromagnetic"
- p1: electron 13 -> 2525. gen: "The photon is the energy carrier of the electromagnetic field. The energy of the photon is given by the"
- p2: electron 9 -> 1178. gen: "The photon, which is the particle of the electromagnetic field, is the carrier of the energy that is"
- p4: electron 12 -> 1122. gen: "The particle that is transferred or shared is the photon, which is the quantum of the electromagnetic field,"
- p5: electron 3 -> 602.  gen: "The photon is the only particle that is massless and has no energy. The photon is the only"
- p7: electron 4 -> 465.  gen: "The photon, the quantum of the electromagnetic field, is the theoretical carrier of the energy and the exchange"

## usage (u) — top token = "Photon" (intact ≈ 1) but degenerate repetition
- p1: gen: "Photon is a photon of light photon is a photon of light photon is a photon of photon is a"
- p2: gen: "Photon. Does a photon have a photonononononononononononononon"
- p4: gen: "Photon. Photon is a photon of light photon is a photon of light photon is a photon of photon"
- p6: gen: "Photon What is the photon? Answer: Photon What is the photon? Answer: Photon"

## baseline (no steering) — coherent electron text, electron rank 3-51
- p1: "An electron absorbs the energy. The electron is a negatively charged particle that orbits the nucleus of an atom"
- p4: "Electrons. What is the energy of the ejected particles proportional to? Answer: The frequency of the"

## Notes
- delta steers toward a physically related particle (photon), not the labelled alternative (proton) — content flip to a related concept.
- Both delta and u move the output toward photon, but delta = coherent content (top = function word, intact low), u = forced bare token (top "Photon", intact≈1, fluency broken).
- Local steering aggregates (e/p, c=16): delta margin_flip=1.00 / intact_flip=0.19; usage margin_flip=1.00 / intact_flip=1.00. The intact metric undercounts the delta content flip.
