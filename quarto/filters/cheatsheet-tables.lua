-- Шпаргалки свёрстаны в две колонки, а longtable (так pandoc выводит таблицы)
-- в двухколоночном режиме не работает. Превращаем таблицы в обычный tabular.
function Table(tbl)
  if not quarto.doc.is_format("latex") then
    return nil
  end
  local latex = pandoc.write(pandoc.Pandoc({ tbl }), "latex")
  latex = latex:gsub("\\begin{longtable}%[%]", "\\begin{tabular}")
  latex = latex:gsub("\\end{longtable}", "\\bottomrule\\noalign{}\n\\end{tabular}")
  latex = latex:gsub("\\endfirsthead.-\\endhead\n", "")
  latex = latex:gsub("\\endhead\n", "")
  latex = latex:gsub("\\bottomrule\\noalign{}\n\\endlastfoot\n", "")
  return pandoc.RawBlock("latex", "\\par\\smallskip\\noindent\\small " .. latex .. "\\par\\smallskip")
end
