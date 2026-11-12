import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'

const CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  toImageButtonOptions: { format: 'png', scale: 2 },
}

export default function Plot({ figure }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current || !figure) return
    const layout = { ...(figure.layout || {}), autosize: true }
    delete layout.width
    Plotly.react(ref.current, figure.data || [], layout, CONFIG)
    const el = ref.current
    const ro = new ResizeObserver(() => Plotly.Plots.resize(el))
    ro.observe(el)
    return () => ro.disconnect()
  }, [figure])
  useEffect(() => {
    const el = ref.current
    return () => el && Plotly.purge(el)
  }, [])
  const h = figure?.layout?.height || 420
  return <div ref={ref} style={{ width: '100%', minHeight: h }} />
}
