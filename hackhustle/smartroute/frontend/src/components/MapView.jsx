import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import maplibregl from 'maplibre-gl'
import { motion } from 'framer-motion'

const RISK_COLORS = { green: '#34d399', yellow: '#fbbf24', red: '#f87171' }
const RISK_LABELS = { green: 'High', yellow: 'Medium', red: 'Low' }
const RISK_BADGE_COLORS = { green: '#059669', yellow: '#d97706', red: '#dc2626' }
const BENGALURU_CENTER = [77.5946, 12.9716]
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'

function createPopupContent(customer, address, riskColor, riskLabel, score, explanation) {
  const container = document.createElement('div')
  container.style.fontFamily = 'Inter, sans-serif'

  const name = document.createElement('p')
  Object.assign(name.style, { fontWeight: '600', fontSize: '13px', margin: '0 0 4px', color: '#111' })
  name.textContent = customer ?? ''
  container.appendChild(name)

  const addr = document.createElement('p')
  Object.assign(addr.style, { fontSize: '11px', color: '#6b7280', margin: '0 0 8px', lineHeight: '1.4' })
  addr.textContent = address ?? ''
  container.appendChild(addr)

  const badge = document.createElement('span')
  Object.assign(badge.style, { background: riskColor, color: 'white', fontSize: '10px', fontWeight: '600', padding: '2px 8px', borderRadius: '99px', display: 'inline-block', marginBottom: '6px' })
  badge.textContent = `${riskLabel} — ${score ?? 0}/100`
  container.appendChild(badge)

  const desc = document.createElement('p')
  Object.assign(desc.style, { fontSize: '10px', color: '#9ca3af', margin: '0', lineHeight: '1.4' })
  desc.textContent = explanation ?? ''
  container.appendChild(desc)

  return container
}

export default function MapView({ orders, loading, viewMode, onViewModeChange, zoomLevel }) {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const popupRef = useRef(null)
  const [mapReady, setMapReady] = useState(false)
  const [mapError, setMapError] = useState(null)
  const handlersRef = useRef([])

  const geojson = useMemo(() => ({
    type: 'FeatureCollection',
    features: (orders || []).map((o) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [o.lng, o.lat] },
      properties: { id: o.id, address: o.address, score: o.score, risk: o.risk, explanation: o.explanation, customer: o.customer_name },
    })),
  }), [orders])

  // Initialize map
  useEffect(() => {
    if (map.current) return
    try {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: MAP_STYLE,
        center: BENGALURU_CENTER,
        zoom: 11,
        pitch: 0,
        attributionControl: false,
      })
      map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left')
      map.current.on('load', () => setMapReady(true))
      map.current.on('error', (e) => console.error('MapLibre error:', e))
    } catch (err) {
      console.error('Failed to initialize map:', err)
      setMapError(err.message || 'Failed to load map')
    }
    return () => { map.current?.remove(); map.current = null }
  }, [])

  // Respond to external zoom level changes
  useEffect(() => {
    if (!map.current || !mapReady || zoomLevel == null) return
    map.current.easeTo({ zoom: zoomLevel, duration: 1200 })
  }, [mapReady, zoomLevel])

  // Update layers when data or viewMode changes
  const updateLayers = useCallback(() => {
    if (!map.current || !mapReady) return

    // Clean up previous handlers
    handlersRef.current.forEach(({ event, layer, handler }) => {
      if (map.current) map.current.off(event, layer, handler)
    })
    handlersRef.current = []

    const on = (event, layer, handler) => {
      map.current.on(event, layer, handler)
      handlersRef.current.push({ event, layer, handler })
    }

    // Remove existing layers/sources
    const layerIds = ['clusters', 'cluster-count', 'unclustered-point', 'raw-points', 'raw-point-border']
    layerIds.forEach((id) => { if (map.current.getLayer(id)) map.current.removeLayer(id) })
    if (map.current.getSource('orders')) map.current.removeSource('orders')
    if (!geojson.features.length) return

    if (viewMode === 'cluster') {
      map.current.addSource('orders', { type: 'geojson', data: geojson, cluster: true, clusterMaxZoom: 14, clusterRadius: 50 })

      map.current.addLayer({
        id: 'clusters', type: 'circle', source: 'orders', filter: ['has', 'point_count'],
        paint: {
          'circle-color': ['step', ['get', 'point_count'], '#99f6e4', 10, '#5eead4', 30, '#14b8a6', 50, '#0d9488'],
          'circle-radius': ['step', ['get', 'point_count'], 20, 10, 28, 30, 36, 50, 44],
          'circle-stroke-width': 3, 'circle-stroke-color': '#ffffff', 'circle-opacity': 0.85,
        },
      })
      map.current.addLayer({
        id: 'cluster-count', type: 'symbol', source: 'orders', filter: ['has', 'point_count'],
        layout: { 'text-field': '{point_count_abbreviated}', 'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'], 'text-size': 13 },
        paint: { 'text-color': '#115e59' },
      })
      map.current.addLayer({
        id: 'unclustered-point', type: 'circle', source: 'orders', filter: ['!', ['has', 'point_count']],
        paint: {
          'circle-color': ['match', ['get', 'risk'], 'green', RISK_COLORS.green, 'yellow', RISK_COLORS.yellow, 'red', RISK_COLORS.red, '#94a3b8'],
          'circle-radius': 7, 'circle-stroke-width': 2, 'circle-stroke-color': '#ffffff',
        },
      })

      on('click', 'clusters', (e) => {
        const features = map.current.queryRenderedFeatures(e.point, { layers: ['clusters'] })
        const clusterId = features[0].properties.cluster_id
        map.current.getSource('orders').getClusterExpansionZoom(clusterId).then((zoom) => {
          map.current.easeTo({ center: features[0].geometry.coordinates, zoom })
        })
      })
      on('click', 'unclustered-point', (e) => {
        const f = e.features[0]
        const { address, score, risk, explanation, customer } = f.properties
        popupRef.current?.remove()
        popupRef.current = new maplibregl.Popup({ offset: 15, maxWidth: '280px' })
          .setLngLat(f.geometry.coordinates)
          .setDOMContent(createPopupContent(customer, address, RISK_BADGE_COLORS[risk] ?? '#6b7280', RISK_LABELS[risk] ?? 'Unknown', score, explanation))
          .addTo(map.current)
      })
      on('mouseenter', 'clusters', () => { map.current.getCanvas().style.cursor = 'pointer' })
      on('mouseleave', 'clusters', () => { map.current.getCanvas().style.cursor = '' })
      on('mouseenter', 'unclustered-point', () => { map.current.getCanvas().style.cursor = 'pointer' })
      on('mouseleave', 'unclustered-point', () => { map.current.getCanvas().style.cursor = '' })
    } else {
      map.current.addSource('orders', { type: 'geojson', data: geojson })
      map.current.addLayer({
        id: 'raw-point-border', type: 'circle', source: 'orders',
        paint: { 'circle-radius': 8, 'circle-color': '#ffffff' },
      })
      map.current.addLayer({
        id: 'raw-points', type: 'circle', source: 'orders',
        paint: {
          'circle-color': ['match', ['get', 'risk'], 'green', RISK_COLORS.green, 'yellow', RISK_COLORS.yellow, 'red', RISK_COLORS.red, '#94a3b8'],
          'circle-radius': 6, 'circle-stroke-width': 2, 'circle-stroke-color': '#ffffff',
        },
      })
      on('click', 'raw-points', (e) => {
        const f = e.features[0]
        const { address, score, risk, explanation, customer } = f.properties
        popupRef.current?.remove()
        popupRef.current = new maplibregl.Popup({ offset: 15, maxWidth: '280px' })
          .setLngLat(f.geometry.coordinates)
          .setDOMContent(createPopupContent(customer, address, RISK_BADGE_COLORS[risk] ?? '#6b7280', RISK_LABELS[risk] ?? 'Unknown', score, explanation))
          .addTo(map.current)
      })
      on('mouseenter', 'raw-points', () => { map.current.getCanvas().style.cursor = 'pointer' })
      on('mouseleave', 'raw-points', () => { map.current.getCanvas().style.cursor = '' })
    }
  }, [mapReady, geojson, viewMode])

  useEffect(() => { updateLayers() }, [updateLayers])

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="bg-white rounded-2xl shadow-card border border-gray-100 overflow-hidden relative hover:shadow-card-hover transition-shadow duration-300"
    >
      {/* View toggle */}
      <div className="absolute top-4 right-4 z-10 flex bg-white/90 backdrop-blur-sm rounded-xl shadow-float border border-gray-100 p-1">
        {['cluster', 'raw'].map((mode) => (
          <button
            key={mode}
            onClick={() => onViewModeChange(mode)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              viewMode === mode ? 'bg-brand-500 text-white shadow-sm' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {mode === 'cluster' ? 'Cluster View' : 'Raw View'}
          </button>
        ))}
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-white/90 backdrop-blur-sm rounded-xl shadow-float border border-gray-100 px-3 py-2">
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> High (80+)</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-400" /> Medium (50-79)</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-red-400" /> Low (&lt;50)</span>
        </div>
      </div>

      {/* Map error */}
      {mapError && (
        <div className="absolute inset-0 z-20 bg-gray-50 flex items-center justify-center">
          <div className="text-center p-6">
            <p className="text-sm font-medium text-gray-600 mb-1">Map unavailable</p>
            <p className="text-xs text-gray-400">{mapError}</p>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="absolute inset-0 z-20 bg-white/80 flex items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <svg className="w-8 h-8 animate-spin text-brand-500" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
              <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" />
            </svg>
            <span className="text-xs text-gray-500">Loading delivery data...</span>
          </div>
        </div>
      )}

      <div ref={mapContainer} className="w-full h-[520px]" />
    </motion.div>
  )
}
