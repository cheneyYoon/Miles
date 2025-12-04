# Project Final Report Handout (64 points)

The final project report is your final deliverable for this project and for the course. Congratulations
on getting here!

How to submit

Any work that is submitted up to 24 hours past the deadline will receive a 20% grade deduction. No
other late work is accepted. Quercus submission time will be used, not your local computer time or
any other screenshots that you provide. You can submit your work as many times as you want before
the deadline, so please submit often and early.

The final report will be submitted as agroupon Quercus by the Final Deliverable deadline. This
document should be written using Latex based on the course template and submitted in PDF format.
There is a4-page limitfor the main text and unlimited references is applied. However, a concisely
written document is preferred. The page limit is hard: A 5-page document will receive a 20% penalty
(in addition to any late penalties), and any document that is longer than 5 pages will receive 0%.

Final Report Rubric

The project final report document is graded out of 64 points.

- Introduction (2 points): A brief description of the motivations behind your project, the goal
    of your project, why it is interesting or important, and why machine learning is a reasonable
    approach.
       ◦2/2 An introduction that clearly describes the project goal, why the project is interesting
          and/or useful, and convincingly describes why machine learning is an appropriate tool
          for the task.
       ◦1/2 The introduction describes the project but is vague or has information that is
          factually incorrect.
       ◦0/2 The introduction does not make it clear what the specific goal of your project is.
- Illustration / Figure (2 points): A figure or a diagram that illustrates the overall model
    or idea of your project. The idea is to make your report more accessible, especially to
    readers who are starting by skimming your work. For the project, taking a picture of a
    hand-drawn diagram is fine, as long as it’s legible. PowerPoint is another option. You will
    not be penalized for hand-drawn illustrations – you are graded on the design and illustrative
    power.
       ◦2/2 A well thought-out figure that communicates the core idea of your project and
          architecture immediately.
       ◦1/2 An illustration that does the job, but is not particularly clear, or possibly too wordy.
       ◦0/2 The illustration is significantly lacking in some respect or contain factual
          inconsistencies or inaccuracies.
- Background & Related Work (2 points): A description of at least 5 related work in the
    field, to provide reader a sense of what has already been done in this area, e.g., papers or
    existing products/software that do a related thing.

```
◦2/2 Briefly describes at least prior work related to your project to put your project into
context. Your descriptions need not be complete but should contain important work.
◦1/2 Background that has omissions or factual incorrectness, but otherwise places your
project into context.
```

```
◦0/2 Background contains too much information not related to your project, or has
major omissions of content, or does not sufficiently put your project into context.
```
- Data Processing (4 points): Describe the data that you have collected and if you have
    preprocessed it in any way. Be clear and specific when describing what you have done,
    so that a classmate can reproduce your work. Show some statistics and examples of your
    data. The extent of data processing will vary from project to project, and you will be graded
    accordingly.

```
◦4/4 Clearly describes sources of data, and the steps you took to clean and format your
data. Statistics and data example are well-chosen and gives readers a “feel” for your
data.
◦3/4 Mostly clear description, but some aspects of the data processing steps are vague.
Statistics and data example are somewhat illustrative/helpful.
◦2/4 Vague description or missing key information about where your data comes from
or what you did. No example data shown, or the ones shown are not illustrative.
◦1/4 Incomplete information.
```
- Data Processing (4 points): Describe the data that you have collected and if you have
    preprocessed it in any way. Be clear and specific when describing what you have done,
    so that a classmate can reproduce your work. Show some statistics and examples of your
    data. The extent of data processing will vary from project to project, and you will be graded
    accordingly.

```
◦4/4 Clearly describes sources of data, and the steps you took to clean and format your
data. Statistics and data example are well-chosen and gives readers a “feel” for your
data.
◦3/4 Mostly clear description, but some aspects of the data processing steps are vague.
Statistics and data example are somewhat illustrative/helpful.
◦2/4 Vague description or missing key information about where your data comes from
or what you did. No example data shown, or the ones shown are not illustrative.
◦1/4 Incomplete information.
```
- Architecture (4 points): A description of the final neural network model architecture. Do
    not describe all the intermediate models that you have tried. Instead, present the model (or
    models) whose quantitative results you will show. These should be your most interesting
    models. Be as specific as you can while being concise. Readers should be able to reproduce
    a model similar enough to yours and obtain a similar performance.
       ◦4/4 Clear and concise description of your model architecture, so that a classmate can
          reproduce a model similar to yours that will perform similarly.
       ◦3/4 Good description of your model architecture, but with either not enough detail to
          be reproducible, or too much unnecessary detail not useful for reproducing your model.
       ◦2/4 Some issues with the description (inconsistencies, factual inaccuracies)
       ◦0/4 Unclear description of the type(s) of neural network model that you will use, or a
          choice that is inconsistent with your problem.
- Baseline Model (4 points): Describe a simple, baseline model that you will compare
    your neural network against. This can be a simple model that you build, a hand-coded
    heuristic model (that does not use machine learning), or a machine learning model with few
    hyperparameters and can be trained quickly.
       ◦4/4 A reasonable choice of baseline, accompanied by a description of the baseline so
          that a knowledgeable classmate can find, reproduce, or build a similar version.
       ◦2/4 An adequate description of a reasonable baseline.
       ◦0/2 Poor choice of baseline inconsistent with the problem.


- Quantitative Results (4 points): A description of the quantitative measures of your result.
    What measurements can you use to illustrate how your model performs?
       ◦4/4 Insightful, well-chosen measurements that illustrate how your model performs.
       ◦3/4 Minor issue with the choice of measurements, or the way the result is presented.
       ◦2/4 Major issue with the choice of measurements, or misleading presentation of the
          results.
       ◦0/4 No result presented.
- Qualitative Results (4 points): Include some sample outputs of your model, to help your
    readers better understand what your model can do. The qualitative results should also put
    your quantitative results into context (e.g., Why did your model perform well? Is there a
    type of input that the model does not do well on?)
       ◦4/4 Insightful, well-chosen outputs that illustrate how your model performs. It is clear
          how you determined which outputs to show, and why.
       ◦3/4 Minor issues with the choice of outputs, or the way the result is presented.
       ◦2/4 Some issues with the choice of outputs, or the way the result is presented.
       ◦0/4 No result presented.
- Evaluate model on new data (10 points): Describe the efforts taken to ensure the results
    are a good representation of the model’s performance on new data. Can you evaluate model
    on new data? This will depend greatly on the problem being solved.
       ◦10/10 Team is able to obtain new samples that have not been examined or used in any
          way to influence the tuning of hyperparameters. Model performance meets or exceeds
             expectations for the problem being solved.
       ◦7/10 Model performance does not meet expectations on new samples.
       ◦4/10 Model performs inconsistently on new samples, but an attempt has been made to
          correctly evaluate the model.
       ◦2/10 Model performance is far below expectations on new samples, but an attempt has
          been made to correctly evaluate the model.
       ◦0/10 No attempt made to evaluate the model on new data.
- Discussion (8 points): Discuss your results. Do you think your model is performing well?
    Why or why not? What is unusual, surprising, or interesting about your results? What did
       you learn?
          ◦8/8 Insightful interpretation of the results that is specific to your project. Exceeds
             expectations.
          ◦6/8 Sound interpretation of the results.
          ◦4/8 Some issues with the interpretation.
          ◦0/8 Discussion does not interpret results, only repeats them.
- Ethical Considerations (2 points): Description of a use of the system that could give rise
    to ethical issues. Are there limitations of your model? Your training data?
       ◦2/2 Thoughtful consideration of ethical issues discussed in class, applied to your model.
       ◦1/2 Some consideration of ethical issues in data collection but missing key elements.
- Project Difficulty / Quality (6 points): A measure of how “difficult” the project is, and
    how well your model performs given the difficulty of your problem. If your problem is more
    difficult than what one might expect, you should clearly articulate why in the body of your
    report.
       ◦6/6 Team creates a model that performs better than expected on a challenging project.
          Team demonstrates learning beyond the requirements of (e.g.,) the labs.


```
◦4/6 Meets the expectations of the difficulty of the project, and the performance looks
adequate. A poor model performance is justified.
◦2/6 Project is “too simple” or does not perform as well as expected.
◦1/6 Below expectations.
```
- Structure, Grammar & Mechanics (8 points): We are looking for a document that is easy
    to follow, grammatically correct, and well-written. The document must be written using
    Latex based on the course template.
       ◦8/8 Clear, concise, and well-written document. Exceeds expectations.
       ◦7/8 Well-written document that could be more concise or less error prone.
       ◦6/8 Well-written document that has some issues with grammar, mechanics, or structure.
          Meets expectations.
       ◦5/8 Reasonably written document with grammar, mechanics, or structural issues.
       ◦4/8 Document has many issues.


