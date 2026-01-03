#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/lovelace:0.3.0": *
#show: codly-init.with()


#let hilcoe-report(
  members: (),
  title: "",
  course-name: "Artificial Intelligence",
  course-code: "CS488",
  batch: "DRB2202",
  section: "A",
  instructor: "",
  term: "Autumn 2025",
  date: datetime.today(),
  show-outline: false,
  body,
) = {
  set page(
    paper: "a4",
    numbering: "1.",
    number-align: center,
    footer: context {
      let page-number = counter(page).get().at(0)
      if page-number > 1 {
        line(length: 100%, stroke: 0.5pt)
        v(-2pt)
        text(size: 12pt, weight: "regular")[
          HiLCoE
          #h(1fr)
          #page-number
          #h(1fr)
          2025
        ]
      }
    },
  )

  set text(
    size: 10pt,
    font: "STIX Two Text",
  )

  show math.equation: set text(font: "STIX Two Math")
  show raw: set text(font: "CMU Typewriter Text")

  set par(justify: true)


  set heading(numbering: "1.")

  align(
    figure(image("images/hilcoe-logo.svg", height: 120pt)),
  )
  // title page
  align(
    center,
    par(spacing: 30pt, text(size: 48pt, tracking: -0.03em, font: "Cooper", [HiLCoE])),
  )

  align(
    center,
    text(size: 18pt, font: "CMU Sans Serif", weight: "medium")[School of Computer Science \ & Technology],
  )

  v(30pt)

  align(center, par(
    leading: 1.2em,
    text(
      size: 40pt,
      font: "CMU Sans Serif",
      tracking: -0.02em,
    )[#emph(title)],
  ))

  v(40pt)


  // meta:
  //
  //
  //
  members = members.sorted()

  let ct = body => text(font: "Inter Display", size: 10pt)[
    #body
  ]


  align(
    center,
    block(
      fill: rgb("#ececfaaa"),
      stroke: 0.5pt,
      inset: (
        right: 36pt,
        left: 36pt,
        top: 20pt,
        bottom: 20pt,
      ),
      table(
        columns: (auto, auto),
        align: (right, left),
        inset: 8pt,
        stroke: none,
        ct()[*Members*],
        for member in members [
          #ct(member) \
        ],

        ct()[*Course*], ct()[#course-name (#course-code)],
        ct()[*Instructor*], ct()[#instructor],
        ct()[*Batch/Section*], ct()[#batch/#section],
        ct()[*Term*], ct()[#term],
        ct()[*Date*], ct()[#date.display("[month repr:short] [day], [year]")],
      ),
    ),
  )

  pagebreak()

  if show-outline {
    outline()

    pagebreak()
  }


  body
}


#show: hilcoe-report.with(
  title: [Group Report \ on \ Machine Learning],
  instructor: "Dr. Seyoum Abebe",
  members: (
    "Alazar Gebremehdin",
    "Hannibal Mussie",
    "Feruz Seid",
    "Yassin Bedru",
    "Samir Bahru",
  ),
)
