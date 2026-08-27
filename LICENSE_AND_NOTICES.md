================================================================================
LICENSE
================================================================================

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the
      purposes of this License, Derivative Works shall not include works
      that remain separable from, or merely link (or bind by name) to the
      interfaces of, the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including the
      original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work.

   4. Redistribution. You may reproduce and distribute copies of the Work
      or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or Derivative
          Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor.

   7. Disclaimer of Warranty. Unless required by applicable law or agreed
      to in writing, Licensor provides the Work on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law or agreed to in writing, shall
      any Contributor be liable to You for damages, including any direct,
      indirect, special, incidental, or consequential damages of any
      character arising as a result of this License or out of the use or
      inability to use the Work.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer, and
      charge a fee for, acceptance of support, warranty, indemnity, or
      other liability obligations consistent with this License.

   END OF TERMS AND CONDITIONS

   Copyright 2026 Siddhant Hitesh Mantri, Anirudh Komanduri, Vishal S,
   Smrutishikta Das, Walunila Aier, Sambhav Jha (Group 10 — DS and AI Lab
   Project, IIT Madras)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


================================================================================
THIRD-PARTY NOTICES
================================================================================

The Apache 2.0 license above covers only the original source code authored
by Group 10 (retrieval pipeline, SQL validation, evaluation scripts,
Streamlit UI, etc.). It does NOT extend to the third-party models,
datasets, and libraries listed below, each of which remains under its own
license and terms of use. Users of this repository must independently
comply with those terms when using, redistributing, or building on this
project.

--- Base Model ---

Qwen3-4B-Instruct-2507 (Alibaba Cloud / Qwen Team)
  Used as the frozen base model for QLoRA fine-tuning.
  Licensed under Apache License 2.0. See the model card on Hugging Face
  for the current license terms: https://huggingface.co/Qwen
  Verify the exact license file shipped with the specific checkpoint you
  pull, as some Qwen variants use a separate Qwen Community License
  rather than Apache 2.0.

QLoRA adapter (checkpoint-375) — original fine-tuned weights produced by
  Group 10, hosted at walz89/checkpoint-375-adapter on Hugging Face.
  As a derivative of the Qwen3 base model, redistribution of these
  weights is subject to the base model's license terms in addition to
  this project's own Apache 2.0 license for the code that produces or
  loads them.

--- Datasets ---

Spider (Yale LILY Lab) — used for primary training and evaluation.
  Released under CC BY-SA 4.0. Attribution required; derivative datasets
  must be shared under the same license. https://yale-lily.github.io/spider

BIRD / BIRD Mini-Dev — used for evaluation and gap analysis.
  Refer to the BIRD benchmark's official license terms for redistribution
  and derivative-use conditions. https://bird-bench.github.io

KaggleDBQA — used for gap analysis.
  Refer to the original KaggleDBQA release for license terms.

No raw dataset files are redistributed in this repository beyond what
each dataset's own license permits; scripts here only consume/process
them.

--- Key Libraries and Frameworks ---

PyTorch                                  — BSD-3-Clause
Hugging Face Transformers / PEFT / Accelerate — Apache License 2.0
bitsandbytes (QLoRA quantization)        — MIT License
FAISS (Meta AI)                          — MIT License
sqlglot                                  — MIT License
Streamlit                                — Apache License 2.0

Version numbers and any additional dependencies are listed in
requirements.txt; each retains the license declared by its own project.

--- Summary Table ---

Component                                | License                        | Notes
------------------------------------------|--------------------------------|------------------
Group 10 original source code             | Apache 2.0                     | See LICENSE section above
Qwen3-4B-Instruct-2507 (base model)        | Apache 2.0 (verify checkpoint) | Third-party
QLoRA adapter weights (checkpoint-375)     | Inherits base model terms      | Third-party derivative
Spider dataset                             | CC BY-SA 4.0                   | Third-party, share-alike
BIRD / BIRD Mini-Dev                       | Per BIRD benchmark terms       | Third-party
KaggleDBQA                                 | Per original release terms     | Third-party
FAISS, sqlglot, bitsandbytes               | MIT                            | Third-party
Transformers, PEFT, Accelerate, Streamlit  | Apache 2.0                     | Third-party
PyTorch                                    | BSD-3-Clause                   | Third-party

If you believe any license above is stated incorrectly or has changed
upstream, please open an issue so it can be corrected.


================================================================================
README SECTION TO ADD (paste this under a "## License" heading in README.md)
================================================================================

## License

The original source code in this repository (retrieval pipeline, SQL
validation, evaluation scripts, Streamlit UI, etc.) is licensed under the
Apache License, Version 2.0 — see the LICENSE section of this file for
the full text.

This license covers code authored by Group 10 only. It does not cover:
- the Qwen3-4B-Instruct-2507 base model or the fine-tuned QLoRA adapter
  weights, which remain subject to their own upstream license, or
- the Spider, BIRD, and KaggleDBQA datasets used for training/evaluation,
  each of which is distributed under its own terms.

See the THIRD-PARTY NOTICES section of this file for a full acknowledgment
of third-party models, datasets, and libraries used in this project,
along with their respective licenses.
