# Resident model lifecycle in the native app

The app keeps a single Model loaded in memory as the resident model while its
process lives, and reuses it across repeated dictations; it reloads only when the
user switches to another Model or on cold start. Chose this because loading a
model file (tens to hundreds of MB) on every inference would make dictation
unbearably slow. This is deliberately distinct from a permanent foreground
service: the Model stays hot in memory even though no service runs when idle, so
we avoid always-on battery drain (see the spec's foreground-service decision).
