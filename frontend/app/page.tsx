'use client'

import { useState, useEffect } from 'react'
import { supabase, type Recommendation, type TopCandidate } from '@/lib/supabase'

interface ResponseMeta {
  videos_analyzed?: number
  from_cache?: boolean
  avg_viral_score?: string
  processing_time_ms?: number
  topic_searched?: string
}

export default function Home() {
  const [topic, setTopic] = useState('')
  const [vibe, setVibe] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState('')
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [topCandidates, setTopCandidates] = useState<TopCandidate[]>([])
  const [meta, setMeta] = useState<ResponseMeta | null>(null)
  const [error, setError] = useState('')
  const [estimatedTime, setEstimatedTime] = useState(30)

  // Loading progress simulation
  useEffect(() => {
    if (!loading) return

    const stages = [
      { time: 0, message: 'Scraping YouTube for trending videos...' },
      { time: 10000, message: 'Analyzing video patterns with AI...' },
      { time: 20000, message: 'Running virality predictions...' },
      { time: 30000, message: 'Generating personalized recommendations...' }
    ]

    const timers = stages.map(stage =>
      setTimeout(() => {
        if (loading) setLoadingProgress(stage.message)
      }, stage.time)
    )

    return () => timers.forEach(clearTimeout)
  }, [loading])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setRecommendations([])
    setTopCandidates([])
    setMeta(null)
    setLoadingProgress('Initializing search...')
    setEstimatedTime(45) // Estimate 45 seconds for dynamic scraping

    const startTime = Date.now()

    try {
      // Call new scrape-and-recommend Edge Function
      const { data, error: funcError } = await supabase.functions.invoke(
        'scrape-and-recommend',
        {
          body: {
            topic,
            vibe: vibe || null,
            userId: null  // TODO: Add auth in future
          }
        }
      )

      if (funcError) {
        console.error('Edge Function error:', funcError)
        throw new Error(funcError.message || 'Failed to generate recommendations')
      }

      if (data.error) {
        throw new Error(data.error)
      }

      const actualTime = Math.round((Date.now() - startTime) / 1000)
      console.log(`Completed in ${actualTime}s, recommendations:`, data)

      setRecommendations(data.recommendations || [])
      setTopCandidates(data.topCandidates || [])
      setMeta(data.meta || null)
    } catch (err: any) {
      console.error('Error:', err)
      setError(err.message || 'Failed to generate recommendations. Please try again.')
    } finally {
      setLoading(false)
      setLoadingProgress('')
    }
  }

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto bg-gradient-to-br from-blue-50 to-purple-50">
      <header className="mb-12 text-center">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Miles AI
        </h1>
        <p className="text-gray-600 text-lg">
          AI-Powered Viral Video Recommendations
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Analyze trending content and get personalized video ideas
        </p>
      </header>

      <form onSubmit={handleSubmit} className="max-w-2xl mx-auto space-y-6 mb-12 bg-white p-8 rounded-2xl shadow-lg">
        <div>
          <label className="block text-sm font-semibold mb-3 text-gray-700">
            What&apos;s your content topic?
          </label>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g., tech reviews, cooking, fitness, travel"
            className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:outline-none transition"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-semibold mb-3 text-gray-700">
            Choose your vibe
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {['funny', 'educational', 'inspirational', 'dramatic'].map(v => (
              <button
                key={v}
                type="button"
                onClick={() => setVibe(v === vibe ? '' : v)}
                className={`p-3 rounded-xl border-2 transition capitalize ${
                  vibe === v
                    ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Analyzing trends...
            </span>
          ) : (
            'Get Viral Ideas'
          )}
        </button>

        {loading && (
          <div className="mt-4 p-5 bg-blue-50 border-2 border-blue-200 rounded-xl">
            <div className="flex items-center space-x-3 mb-3">
              <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full"/>
              <p className="text-blue-800 font-medium">{loadingProgress}</p>
            </div>
            <div className="flex items-center justify-between text-sm text-blue-600">
              <span>⏱️ Estimated time: ~{estimatedTime}s</span>
              <span className="text-xs">First request may take longer (cold start)</span>
            </div>
            <div className="mt-3 text-xs text-blue-500">
              💡 Tip: Results are cached for 6 hours - repeated searches will be instant!
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 border-2 border-red-200 rounded-xl text-red-700">
            <p className="font-semibold mb-1">❌ Error</p>
            {error}
          </div>
        )}
      </form>

      {recommendations.length > 0 && (
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Metadata Banner */}
          {meta && (
            <div className="bg-gradient-to-r from-green-50 to-blue-50 border-2 border-green-200 rounded-xl p-4">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div className="flex items-center space-x-2">
                  {meta.from_cache ? (
                    <>
                      <span className="text-2xl">⚡</span>
                      <div>
                        <p className="text-sm font-semibold text-green-800">Instant Results (Cached)</p>
                        <p className="text-xs text-green-600">Data from recent search</p>
                      </div>
                    </>
                  ) : (
                    <>
                      <span className="text-2xl">🔥</span>
                      <div>
                        <p className="text-sm font-semibold text-blue-800">Fresh Data</p>
                        <p className="text-xs text-blue-600">Just scraped from YouTube</p>
                      </div>
                    </>
                  )}
                </div>
                <div className="flex items-center gap-4 text-xs text-gray-600">
                  <span>📊 {meta.videos_analyzed || 0} videos analyzed</span>
                  <span>⭐ {(parseFloat(meta.avg_viral_score || '0') * 100).toFixed(0)}% avg viral score</span>
                  <span>⏱️ {((meta.processing_time_ms || 0) / 1000).toFixed(1)}s</span>
                </div>
              </div>
            </div>
          )}

          <section>
            <h2 className="text-3xl font-bold mb-6 text-gray-800">
              Your Viral Video Ideas
            </h2>
            <div className="space-y-4">
              {recommendations.map((rec, idx) => (
                <div key={idx} className="p-6 bg-white rounded-2xl shadow-md hover:shadow-xl transition">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl flex items-center justify-center text-white font-bold text-xl">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-bold text-xl mb-2 text-gray-800">
                        {rec.title}
                      </h3>
                      <p className="text-gray-600 leading-relaxed">
                        {rec.reasoning}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {topCandidates.length > 0 && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-gray-800">
                Inspired by these trending videos
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                {topCandidates.map((video, idx) => (
                  <div key={idx} className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition">
                    <img
                      src={video.thumbnail_url}
                      alt={video.title}
                      className="w-full h-48 object-cover"
                    />
                    <div className="p-4">
                      <p className="font-semibold text-sm mb-2 line-clamp-2">
                        {video.title}
                      </p>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>{(video.view_count / 1000).toFixed(0)}K views</span>
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full font-semibold">
                          {(video.miles_score * 100).toFixed(0)}% viral
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </main>
  )
}
