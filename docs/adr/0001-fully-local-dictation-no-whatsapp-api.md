# Fully local dictation, no WhatsApp API

We build dictation that runs 100% on-device (faster-whisper on CPU) and never
touches the WhatsApp API. Chose this to guarantee privacy, zero cost, and no
account-ban risk. A WhatsApp API would enable automessaging but requires business
verification/fees (official) or risks account bans (unofficial libraries); both are
explicitly deferred.
