Subject: Original tokenizer and provenance for n0w0f/MatText-slices-2m

To: Nawaf Alampara, Santiago Miret, and Kevin Maik Jablonka (MatText authors)
Status: DRAFT — NOT SENT

Dear MatText authors,

I am developing a thesis on transfer learning for ABS3 chalcogenides and would like to reproduce MatText tokenization faithfully before feature extraction or supervised fine-tuning.

The checkpoint is n0w0f/MatText-slices-2m, HF revision `c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e`, model.safetensors SHA-256 `6bef5652d758190e6e6eaf85ccf2a2f584b69255f1be3bb7bdd72cb1f12a74bd`. Its config references `santiago_ckpt_normal/slice_pt_2m/checkpoint-393000` and declares vocab_size30522 and pad_token_id0.

Could you please provide or confirm:

1. The original tokenizer artifacts used for this exact checkpoint.
2. The complete ordered vocabulary and exact token-to-ID mapping.
3. CLS, SEP, PAD, MASK and UNK IDs, plus any other special tokens.
4. Padding/truncation settings and attention/special-token mask construction, including the MLM collator/runtime versions.
5. Whether the effective vocabulary was smaller than30522 and the model inherited a generic BERT config.
6. How pad_token_id0 relates to the public SliceTokenizer token `o o o` at ID0.
7. The pretraining code commits, effective run configuration/overrides, corpus revision and encoder export procedure.
8. One original SLICES string with its exact input_ids and attention_mask, including padding if used.
9. If available, its CLS output from this exact encoder in eval mode, with dtype/version and numerical tolerance.
10. Whether this public checkpoint corresponds to the BERT protocol evaluated in your Nature Computational Science paper (doi:10.1038/s43588-026-01045-2).

The pre-upload historical source reproduces the documented Ga/P example. The archived pretraining special-token order instead gives UNK149/PAD150/CLS151/SEP152/MASK153. We cannot yet bind that configuration to the exact weights. We also observed public source/runtime discrepancies described in our audit; an original encoded example and effective configuration would help resolve them without changing the model.

Thank you for your help with scientific reproducibility.

Best regards,
[Name / institution]
