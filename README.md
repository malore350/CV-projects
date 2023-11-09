---
abstract: |
  This report presents a comprehensive analysis of established machine
  learning techniques applied to computer vision tasks as outlined in
  the coursework. We systematically address the implementation and
  evaluation of Eigenfaces using Principal Component Analysis (PCA),
  Incremental PCA, and Linear Discriminant Analysis (LDA) for face
  recognition. The effectiveness of ensemble learning through LDA and
  the optimization of Random Forest classifiers are also explored. Each
  method's performance is critically assessed by examining accuracy,
  error rates, and computational efficiency. The findings contribute to
  a deeper understanding of the practical applications of these
  algorithms in the field of computer vision.
author:
- Kamran Gasimov$^{1}$ and Alexandra Suchshenko$^{2}$ [^1] [^2]
title: " **CS485 Machine Learning for Computer Vision: Coursework 1** "
---

# NAIVE EIGENFACE APPROACH

## Approach

The approach employed throughout this work, Eigenface approach, treats
the images as linearized vector of an image, which takes $D\times1$
space, where $D = W\times H$. A method to distill the essential
information from a face image involves identifying the primary axes of
variation within a set of face images and using these axes to encode and
differentiate between individual faces. Face images are mapped onto a
feature space defined by the most significant variations across the set
of images. These pivotal features are termed eigenfaces, which are
essentially the principal components or eigenvectors of the face
dataset. These eigenfaces might not align with identifiable facial
features like eyes or mouth. To recognize a face, a new image is
projected onto the space defined by the eigenfaces, and its identity is
determined by comparing its coordinates in this \"face space\" with
those of known faces.

## Dataset

The experiment utilized a dataset comprising 520 facial images, each
with dimensions of 46 by 56 pixels (which makes number of features of
every vector to be 2576), categorized into 52 distinct classes with 10
images per class. The dataset was divided into training and testing sets
with a ratio of 8:2, resulting in 416 images for training and 104 for
testing. Data has been divided in a way train/test datasets do not share
a common class, to remove bias in a data. Normalization of the training
images was achieved by deducting the average face image, calculated as
the mean across all images, from each individual image. This process
involved no additional data augmentation or transformation.

# GENERATIVE AND DISCRIMINATIVE SUBSPACE LEARNING

We had already discussed the PCA's ability to reconstruct data and LDA's
ability to extract discriminative features. The subspace learning that
would include both would need to have an objective function that would
balance them, and give the power to control the weight and preference.
Here, we will exlain the math behind such an objective function.

To learn the subspace for reconstruction and As discussed earlier, the
objective function of PCA is to achieve the maximum variance in data.
Assuming that the direction of space is defined by $w\in R^D$, the mean
is $\mu = w^T \bar{x}$, which is a mean of projections of every data
point $x_n$ onto $w$. By definition, variance ($\delta^2$) =
$\frac{1}{N} \sum\limits_{n=1}^{N} (x - \mu)^2$. Thus, the variance
becomes:
$$\delta^2 = \frac{1}{N} \sum\limits_{n=1}^{N} (w^T x_n -w^T \bar{x})^2 = \frac{1}{N} \sum\limits_{n=1}^{N} (w^T(x_n - \bar{x}))^2$$
By doing expanding and putting $w$ and $w^T$ out of the sum, we can
achieve the following:
$$\delta^2 = w^T (\frac{1}{N} \sum\limits_{n=1}^{N} (x_n - \bar{x}) (x_n - \bar{x})^T) w$$
where
$\frac{1}{N} \sum\limits_{n=1}^{N} (x_n - \bar{x}) (x_n - \bar{x})^T$ is
a covariance matrix *S*. Thus, our objective function becomes the
following: $$J_{PCA}(w) = w^T S w$$

The objective function for LDA is the Fisher's criterion, known as
generalized Rayleigh quotient:
$$J_{LDA}(w) = \frac{w^T S_B w}{w^T S_W w}$$ where *$S_B, S_W$* are
between-class and within-class scatter matrices, respectively. Then, we
can combine the objective functions using a weighted sum approach:
$$\underset{w}{max} \hspace{0.1cm} J(w) = \alpha J_{LDA}(w) + (1 - \alpha) J_{PCA}(w) =$$
$$=\alpha \frac{w^T S_B w}{w^T S_W w} + (1 - \alpha) w^TSw$$ And the
corresponding Lagrangian for this function will be
$$\mathcal{L}(w, \lambda) = \alpha w^TS_Bw + \lambda(\alpha(k - w^TS_Ww)) + (1-\alpha)w^TSw$$
Setting $\frac{\partial\mathcal{L}}{\partial w} = 0$ and simplifying, we
get:
$$\frac{\partial\mathcal{L}}{\partial w} = \alpha S_Bw - \lambda\alpha S_Ww + (1-\alpha)Sw = 0$$

# PROCEDURE FOR PAPER SUBMISSION

## Selecting a Template (Heading 2)

First, confirm that you have the correct template for your paper size.
This template has been tailored for output on the US-letter paper size.
Please do not use it for A4 paper since the margin requirements for A4
papers may be different from Letter paper size.

## Maintaining the Integrity of the Specifications

The template is used to format your paper and style the text. All
margins, column widths, line spaces, and text fonts are prescribed;
please do not alter them. You may note peculiarities. For example, the
head margin in this template measures proportionately more than is
customary. This measurement and others are deliberate, using
specifications that anticipate your paper as one part of the entire
proceedings, and not as an independent document. Please do not revise
any of the current designations

# MATH

Before you begin to format your paper, first write and save the content
as a separate text file. Keep your text and graphic files separate until
after the text has been formatted and styled. Do not use hard tabs, and
limit use of hard returns to only one return at the end of a paragraph.
Do not add any kind of pagination anywhere in the paper. Do not number
text heads-the template will do that for you.

Finally, complete content and organizational editing before formatting.
Please take note of the following items when proofreading spelling and
grammar:

## Abbreviations and Acronyms

Define abbreviations and acronyms the first time they are used in the
text, even after they have been defined in the abstract. Abbreviations
such as IEEE, SI, MKS, CGS, sc, dc, and rms do not have to be defined.
Do not use abbreviations in the title or heads unless they are
unavoidable.

## Units

-   Use either SI (MKS) or CGS as primary units. (SI units are
    encouraged.) English units may be used as secondary units (in
    parentheses). An exception would be the use of English units as
    identifiers in trade, such as "3.5-inch disk drive".

-   Avoid combining SI and CGS units, such as current in amperes and
    magnetic field in oersteds. This often leads to confusion because
    equations do not balance dimensionally. If you must use mixed units,
    clearly state the units for each quantity that you use in an
    equation.

-   Do not mix complete spellings and abbreviations of units: "Wb/m2" or
    "webers per square meter", not "webers/m2". Spell out units when
    they appear in text: "...a few henries", not "...a few H".

-   Use a zero before decimal points: "0.25", not ".25". Use "cm$^3$",
    not "cc". (bullet list)

## Equations

The equations are an exception to the prescribed specifications of this
template. You will need to determine whether or not your equation should
be typed using either the Times New Roman or the Symbol font (please no
other font). To create multileveled equations, it may be necessary to
treat the equation as a graphic and insert it into the text after your
paper is styled. Number equations consecutively. Equation numbers,
within parentheses, are to position flush right, as in (1), using a
right tab stop. To make your equations more compact, you may use the
solidus ( / ), the exp function, or appropriate exponents. Italicize
Roman symbols for quantities and variables, but not Greek symbols. Use a
long dash rather than a hyphen for a minus sign. Punctuate equations
with commas or periods when they are part of a sentence, as in
$$\alpha + \beta = \chi$$

Note that the equation is centered using a center tab stop. Be sure that
the symbols in your equation have been defined before or immediately
following the equation. Use "(1)", not "Eq. (1)" or "equation (1)",
except at the beginning of a sentence: "Equation (1) is..."

## Some Common Mistakes

-   The word "data" is plural, not singular.

-   The subscript for the permeability of vacuum ?0, and other common
    scientific constants, is zero with subscript formatting, not a
    lowercase letter "o".

-   In American English, commas, semi-/colons, periods, question and
    exclamation marks are located within quotation marks only when a
    complete thought or name is cited, such as a title or full
    quotation. When quotation marks are used, instead of a bold or
    italic typeface, to highlight a word or phrase, punctuation should
    appear outside of the quotation marks. A parenthetical phrase or
    statement at the end of a sentence is punctuated outside of the
    closing parenthesis (like this). (A parenthetical sentence is
    punctuated within the parentheses.)

-   A graph within a graph is an "inset", not an "insert". The word
    alternatively is preferred to the word "alternately" (unless you
    really mean something that alternates).

-   Do not use the word "essentially" to mean "approximately" or
    "effectively".

-   In your paper title, if the words "that uses" can accurately replace
    the word "using", capitalize the "u"; if not, keep using
    lower-cased.

-   Be aware of the different meanings of the homophones "affect" and
    "effect", "complement" and "compliment", "discreet" and "discrete",
    "principal" and "principle".

-   Do not confuse "imply" and "infer".

-   The prefix "non" is not a word; it should be joined to the word it
    modifies, usually without a hyphen.

-   There is no period after the "et" in the Latin abbreviation "et
    al.".

-   The abbreviation "i.e." means "that is", and the abbreviation "e.g."
    means "for example".

# USING THE TEMPLATE

Use this sample document as your LaTeX source file to create your
document. Save this file as **root.tex**. You have to make sure to use
the cls file that came with this distribution. If you use a different
style file, you cannot expect to get required margins. Note also that
when you are creating your out PDF file, the source file is only part of
the equation. *Your TeX $\rightarrow$ PDF filter determines the output
file size. Even if you make all the specifications to output a letter
file in the source - if you filter is set to produce A4, you will only
get A4 output.*

It is impossible to account for all possible situation, one would
encounter using TeX. If you are using multiple TeX files you must make
sure that the "MAIN" source file is called root.tex - this is
particularly important if your conference is using PaperPlaza's built in
TeX to PDF conversion tool.

## Headings, etc

Text heads organize the topics on a relational, hierarchical basis. For
example, the paper title is the primary text head because all subsequent
material relates and elaborates on this one topic. If there are two or
more sub-topics, the next level head (uppercase Roman numerals) should
be used and, conversely, if there are not at least two sub-topics, then
no subheads should be introduced. Styles named "Heading 1", "Heading 2",
"Heading 3", and "Heading 4" are prescribed.

## Figures and Tables

Positioning Figures and Tables: Place figures and tables at the top and
bottom of columns. Avoid placing them in the middle of columns. Large
figures and tables may span across both columns. Figure captions should
be below the figures; table heads should appear above the tables. Insert
figures and tables after they are cited in the text. Use the
abbreviation "Fig. 1", even at the beginning of a sentence.

::: center
::: {#table_example}
    One    Two
  ------- ------
   Three   Four

  : An Example of a Table
:::
:::

Figure Labels: Use 8 point Times New Roman for Figure labels. Use words
rather than symbols or abbreviations when writing Figure axis labels to
avoid confusing the reader. As an example, write the quantity
"Magnetization", or "Magnetization, M", not just "M". If including units
in the label, present them within parentheses. Do not label axes only
with units. In the example, write "Magnetization (A/m)" or
"Magnetization A\[m(1)\]", not just "A/m". Do not label axes with a
ratio of quantities and units. For example, write "Temperature (K)", not
"Temperature/K."

# CONCLUSIONS

A conclusion section is not required. Although a conclusion may review
the main points of the paper, do not replicate the abstract as the
conclusion. A conclusion might elaborate on the importance of the work
or suggest applications and extensions.

# APPENDIX {#appendix .unnumbered}

Appendixes should appear before the acknowledgment.

# ACKNOWLEDGMENT {#acknowledgment .unnumbered}

The preferred spelling of the word "acknowledgment" in America is
without an "e" after the "g". Avoid the stilted expression, "One of us
(R. B. G.) thanks . . ." Instead, try "R. B. G. thanks". Put sponsor
acknowledgments in the unnumbered footnote on the first page.

References are important to the reader; therefore, each citation must be
complete and correct. If at all possible, references should be commonly
available publications.

::: thebibliography
99

G. O. Young, "Synthetic structure of industrial plastics (Book style
with paper title and editor)," in Plastics, 2nd ed. vol. 3, J. Peters,
Ed. New York: McGraw-Hill, 1964, pp. 15--64. W.-K. Chen, Linear Networks
and Systems (Book style). Belmont, CA: Wadsworth, 1993, pp. 123--135. H.
Poor, An Introduction to Signal Detection and Estimation. New York:
Springer-Verlag, 1985, ch. 4. B. Smith, "An approach to graphs of linear
forms (Unpublished work style)," unpublished. E. H. Miller, "A note on
reflector arrays (Periodical styleÑAccepted for publication)," IEEE
Trans. Antennas Propagat., to be publised. J. Wang, "Fundamentals of
erbium-doped fiber amplifiers arrays (Periodical styleÑSubmitted for
publication)," IEEE J. Quantum Electron., submitted for publication. C.
J. Kaufman, Rocky Mountain Research Lab., Boulder, CO, private
communication, May 1995. Y. Yorozu, M. Hirano, K. Oka, and Y. Tagawa,
"Electron spectroscopy studies on magneto-optical media and plastic
substrate interfaces(Translation Journals style)," IEEE Transl. J.
Magn.Jpn., vol. 2, Aug. 1987, pp. 740--741 \[Dig. 9th Annu. Conf.
Magnetics Japan, 1982, p. 301\]. M. Young, The Techincal Writers
Handbook. Mill Valley, CA: University Science, 1989. J. U. Duncombe,
"Infrared navigationÑPart I: An assessment of feasibility (Periodical
style)," IEEE Trans. Electron Devices, vol. ED-11, pp. 34--39, Jan.
1959. S. Chen, B. Mulgrew, and P. M. Grant, "A clustering technique for
digital communications channel equalization using radial basis function
networks," IEEE Trans. Neural Networks, vol. 4, pp. 570--578, July 1993.
R. W. Lucky, "Automatic equalization for digital communication," Bell
Syst. Tech. J., vol. 44, no. 4, pp. 547--588, Apr. 1965. S. P. Bingulac,
"On the compatibility of adaptive controllers (Published Conference
Proceedings style)," in Proc. 4th Annu. Allerton Conf. Circuits and
Systems Theory, New York, 1994, pp. 8--16. G. R. Faulhaber, "Design of
service systems with priority reservation," in Conf. Rec. 1995 IEEE Int.
Conf. Communications, pp. 3--8. W. D. Doyle, "Magnetization reversal in
films with biaxial anisotropy," in 1987 Proc. INTERMAG Conf., pp.
2.2-1--2.2-6. G. W. Juette and L. E. Zeffanella, "Radio noise currents n
short sections on bundle conductors (Presented Conference Paper style),"
presented at the IEEE Summer power Meeting, Dallas, TX, June 22--27,
1990, Paper 90 SM 690-0 PWRS. J. G. Kreifeldt, "An analysis of
surface-detected EMG as an amplitude-modulated noise," presented at the
1989 Int. Conf. Medicine and Biological Engineering, Chicago, IL. J.
Williams, "Narrow-band analyzer (Thesis or Dissertation style)," Ph.D.
dissertation, Dept. Elect. Eng., Harvard Univ., Cambridge, MA, 1993. N.
Kawasaki, "Parametric study of thermal and chemical nonequilibrium
nozzle flow," M.S. thesis, Dept. Electron. Eng., Osaka Univ., Osaka,
Japan, 1993. J. P. Wilkinson, "Nonlinear resonant circuit devices
(Patent style)," U.S. Patent 3 624 12, July 16, 1990.
:::

[^1]: $^{1}$Department of Aerospace Engineering, KAIST,
    `kamran.qasimoff at kaist.ac.kr`

[^2]: $^{2}$Department of Aerospace Engineering, KAIST,
    `alexandra.suchshenko at kaist.ac.kr`
