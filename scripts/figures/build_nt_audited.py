def main():
    import argparse

    argparse.ArgumentParser(description="build nt audited").parse_args()
    """Create an editable, audited NT diagram and derive PDF/SVG/PNG from its XML."""
    import html
    import xml.etree.ElementTree as E
    from pathlib import Path

    import pypdfium2 as pdfium
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas

    OUT = Path(__file__).parent / "audited_20260911"
    OUT.mkdir(exist_ok=True)
    W, H = 900, 930
    mx = E.Element("mxfile", host="app.diagrams.net")
    d = E.SubElement(mx, "diagram", id="nt-audited", name="Nucleotide Transformer")
    g = E.SubElement(
        d, "mxGraphModel", page="1", pageWidth=str(W), pageHeight=str(H), grid="1", gridSize="10"
    )
    root = E.SubElement(g, "root")
    E.SubElement(root, "mxCell", id="0")
    E.SubElement(root, "mxCell", id="1", parent="0")

    def box(id, x, y, w, h, lines, fill="#EAF0F6", size=20):
        cell = E.SubElement(
            root,
            "mxCell",
            id=id,
            parent="1",
            vertex="1",
            value="\n".join(lines),
            style=f"rounded=1;arcSize=8;whiteSpace=wrap;html=0;fillColor={fill};strokeColor=#63758A;strokeWidth=1.5;fontFamily=Arial;fontSize={size};align=center;verticalAlign=middle;",
        )
        E.SubElement(
            cell,
            "mxGeometry",
            x=str(x),
            y=str(y),
            width=str(w),
            height=str(h),
            attrib={"as": "geometry"},
        )

    def edge(id, source, target, pts):
        c = E.SubElement(
            root,
            "mxCell",
            id=id,
            parent="1",
            edge="1",
            source=source,
            target=target,
            value="",
            style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#263238;strokeWidth=2;endArrow=block;endFill=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",
        )
        geom = E.SubElement(c, "mxGeometry", relative="1", attrib={"as": "geometry"})
        arr = E.SubElement(geom, "Array", attrib={"as": "points"})
        for x, y in pts:
            E.SubElement(arr, "mxPoint", x=str(x), y=str(y))

    box(
        "title",
        40,
        20,
        820,
        55,
        ["Nucleotide Transformer: frozen features and IA3"],
        fill="#FFFFFF",
        size=26,
    )
    box(
        "tokens",
        40,
        100,
        820,
        110,
        [
            "201-bp DNA: pretrained tokenizer produces token IDs + mask",
            "Non-overlapping 6-mers; residual bases use single-base tokens",
            "37 tokens for an unambiguous 201-bp input, including <cls>",
        ],
        fill="#E8F2F5",
        size=20,
    )
    box(
        "frozen",
        40,
        275,
        390,
        250,
        [
            "FROZEN FEATURE ROUTE",
            "NT-v2-50M: all backbone weights fixed",
            "12 blocks; hidden width 512; 16 heads",
            "Rotary attention; gated FFN width 2048",
            "",
            "Primary analysis: final-layer features",
            "Secondary: layer 6 or learned layer mixture",
        ],
        size=19,
    )
    box(
        "ia3",
        470,
        275,
        390,
        250,
        [
            "IA3 ADAPTATION ROUTE (post hoc)",
            "Same NT backbone; original weights fixed",
            "Learn scales inside each of 12 blocks:",
            "keys: 512; values: 512; FFN output: 2048",
            "Scales start at 1 and are unconstrained",
            "",
            "36,864 adapter parameters",
            "Use final-layer features",
        ],
        fill="#F4EEDC",
        size=19,
    )
    box(
        "head",
        40,
        595,
        820,
        160,
        [
            "SAME HEAD ARCHITECTURE (fitted separately)",
            "Token projection: 512 -> 256, GELU, LayerNorm",
            "Masked attention-weighted mean (256) + masked maximum (256)",
            "Concatenate: 512 features -> MLP 512 -> 256 -> 128 -> 20",
            "MLP hidden layers: GELU and dropout 0.1",
        ],
        fill="#F4EEDC",
        size=20,
    )
    box(
        "out",
        40,
        805,
        820,
        60,
        ["20-bin accessibility profile (linear output)"],
        fill="#E5F1E3",
        size=23,
    )
    box(
        "note",
        40,
        883,
        820,
        30,
        ["Alternative model routes; blue = frozen backbone, ochre = trainable components"],
        fill="#FFFFFF",
        size=17,
    )
    edge("to_frozen", "tokens", "frozen", [(450, 210), (450, 242), (235, 242), (235, 275)])
    edge("to_ia3", "tokens", "ia3", [(450, 210), (450, 242), (665, 242), (665, 275)])
    edge("from_frozen", "frozen", "head", [(235, 525), (235, 560), (450, 560), (450, 595)])
    edge("from_ia3", "ia3", "head", [(665, 525), (665, 560), (450, 560), (450, 595)])
    edge("prediction", "head", "out", [(450, 755), (450, 805)])
    source = OUT / "nt_architecture_audited.drawio"
    E.indent(mx)
    E.ElementTree(mx).write(source, encoding="utf-8", xml_declaration=True)

    # Export a deliberately small subset of draw.io primitives from the saved XML.
    cells = E.parse(source).findall(".//mxCell")
    c = canvas.Canvas(str(OUT / "nt_architecture_audited.pdf"), pagesize=(W, H))
    c.setTitle("Audited Nucleotide Transformer architecture")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for cell in cells:
        if cell.get("vertex") != "1":
            continue
        geo = cell.find("mxGeometry")
        x, y, w, h = [float(geo.get(k)) for k in ("x", "y", "width", "height")]
        style = dict(s.split("=", 1) for s in cell.get("style").split(";") if "=" in s)
        fill = style["fillColor"]
        size = float(style["fontSize"])
        stroke = fill if cell.get("id") in ("title", "note") else "#63758A"
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(stroke))
        c.setLineWidth(1.5)
        c.roundRect(x, H - y - h, w, h, 8, fill=1, stroke=1)
        svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        lines = cell.get("value").splitlines()
        lineh = size * 1.4
        start = y + (h - (len(lines) - 1) * lineh) / 2 + size * 0.34
        for j, line in enumerate(lines):
            yy = start + j * lineh
            bold = j == 0 and cell.get("id") not in ("tokens", "note")
            font = "Helvetica-Bold" if bold else "Helvetica"
            c.setFont(font, size)
            c.setFillColor(HexColor("#263238"))
            assert c.stringWidth(line, font, size) < w - 16, (cell.get("id"), line)
            c.drawCentredString(x + w / 2, H - yy, line)
            svg.append(
                f'<text x="{x + w / 2}" y="{yy}" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="#263238">{html.escape(line)}</text>'
            )
    for cell in cells:
        if cell.get("edge") != "1":
            continue
        pts = [(float(p.get("x")), float(p.get("y"))) for p in cell.findall(".//mxPoint")]
        c.setStrokeColor(HexColor("#263238"))
        c.setFillColor(HexColor("#263238"))
        c.setLineWidth(2)
        p = c.beginPath()
        p.moveTo(pts[0][0], H - pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, H - y)
        c.drawPath(p)
        x, y = pts[-1]
        p = c.beginPath()
        p.moveTo(x, H - y)
        p.lineTo(x - 5, H - y + 9)
        p.lineTo(x + 5, H - y + 9)
        p.close()
        c.drawPath(p, fill=1)
        svg.append(
            '<polyline points="'
            + " ".join(f"{x},{y}" for x, y in pts)
            + '" fill="none" stroke="#263238" stroke-width="2"/>'
        )
        svg.append(f'<polygon points="{x},{y} {x - 5},{y - 9} {x + 5},{y - 9}" fill="#263238"/>')
    c.showPage()
    c.save()
    svg.append("</svg>")
    (OUT / "nt_architecture_audited.svg").write_text("\n".join(svg), encoding="utf-8")
    pdf = pdfium.PdfDocument(OUT / "nt_architecture_audited.pdf")
    pdf[0].render(scale=1.25).to_pil().save(OUT / "nt_architecture_audited.png")
    print(source)


if __name__ == "__main__":
    main()
