// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts"
import { createClient } from 'jsr:@supabase/supabase-js@2'
import Anthropic from 'npm:@anthropic-ai/sdk@0.32.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  const startTime = Date.now()

  try {
    const { topic, vibe, userId } = await req.json()
    console.log(`Processing dynamic scrape request: topic="${topic}", vibe="${vibe}"`)

    // Initialize Supabase client
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // STEP 1: Check cache first (avoid re-scraping same topic within 6 hours)
    const cacheKey = topic.toLowerCase().trim()
    const cacheExpiry = new Date(Date.now() - 6 * 60 * 60 * 1000) // 6 hours ago

    console.log('Checking cache for topic:', cacheKey)
    const { data: cachedData, error: cacheError } = await supabaseClient
      .from('scrape_cache')
      .select('*')
      .eq('topic', cacheKey)
      .gte('created_at', cacheExpiry.toISOString())
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle()

    let scrapedVideos: any[] = []
    let fromCache = false

    if (cachedData && cachedData.video_data) {
      console.log('✓ Using cached scrape results')
      scrapedVideos = cachedData.video_data as any[]
      fromCache = true
    } else {
      console.log('✗ No cache found, initiating dynamic scraping...')

      // STEP 2: Call scraper API to get fresh videos
      const scraperUrl = Deno.env.get('SCRAPER_API_URL') || 'http://localhost:8001'
      console.log(`Calling scraper API: ${scraperUrl}/scrape`)

      try {
        const scrapeResponse = await fetch(`${scraperUrl}/scrape`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic: topic,
            max_results: 20,
            max_duration: 60
          })
        })

        if (!scrapeResponse.ok) {
          const errorText = await scrapeResponse.text()
          console.error('Scraper API error:', errorText)

          // Check if it's a YouTube bot detection issue
          if (errorText.includes('Sign in to confirm') || errorText.includes('not a bot')) {
            return new Response(
              JSON.stringify({
                error: 'YouTube temporarily blocked scraping. Please try again in a few minutes or try a different topic.',
                details: 'Our scraper was flagged as a bot. This happens occasionally on free hosting.',
                suggestion: 'Try: a different topic, or wait 2-3 minutes and retry'
              }),
              { status: 503, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
            )
          }

          throw new Error(`Scraper API failed (${scrapeResponse.status}): ${errorText}`)
        }

        const scrapeData = await scrapeResponse.json()
        console.log(`✓ Scraped ${scrapeData.videos_found} videos`)

        if (!scrapeData.success || !scrapeData.videos || scrapeData.videos.length === 0) {
          return new Response(
            JSON.stringify({
              error: 'No videos found for this topic. Try a different search term.',
              topic: topic
            }),
            { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
          )
        }

        // STEP 3: Run inference on each scraped video
        const inferenceUrl = Deno.env.get('INFERENCE_API_URL') || 'https://cheneyyoon-miles-inference.hf.space'
        console.log(`Running inference on ${scrapeData.videos.length} videos...`)

        const inferencePromises = scrapeData.videos.map(async (video: any) => {
          try {
            const inferenceResponse = await fetch(`${inferenceUrl}/predict`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                title: video.title,
                description: video.description || '',
                thumbnail_url: video.thumbnail_url,
                view_count: video.view_count || 0,
                like_count: video.like_count || 0,
                comment_count: video.comment_count || 0,
                duration_seconds: video.duration_seconds || 30
              })
            })

            if (!inferenceResponse.ok) {
              console.error(`Inference failed for video ${video.video_id}`)
              return {
                ...video,
                miles_score: 0.5, // Default score if inference fails
                predicted_velocity: 0
              }
            }

            const prediction = await inferenceResponse.json()
            return {
              ...video,
              miles_score: prediction.viral_score,
              predicted_velocity: prediction.predicted_velocity
            }
          } catch (error) {
            console.error(`Error running inference for ${video.video_id}:`, error)
            return {
              ...video,
              miles_score: 0.5,
              predicted_velocity: 0
            }
          }
        })

        scrapedVideos = await Promise.all(inferencePromises)
        console.log(`✓ Inference complete. Avg score: ${(scrapedVideos.reduce((sum, v) => sum + v.miles_score, 0) / scrapedVideos.length).toFixed(3)}`)

        // STEP 4: Cache the results for future requests
        try {
          await supabaseClient.from('scrape_cache').insert({
            topic: cacheKey,
            video_data: scrapedVideos,
            video_count: scrapedVideos.length
          })
          console.log('✓ Results cached')
        } catch (cacheInsertError) {
          console.error('Failed to cache results (non-fatal):', cacheInsertError)
        }
      } catch (scrapeError) {
        console.error('Scraping/inference error:', scrapeError)
        return new Response(
          JSON.stringify({
            error: 'Failed to scrape videos for this topic',
            details: scrapeError.message
          }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }
    }

    // STEP 5: Filter and sort top performers
    const topCandidates = scrapedVideos
      .filter(v => v.miles_score > 0)
      .sort((a, b) => b.miles_score - a.miles_score)
      .slice(0, 10)

    if (topCandidates.length === 0) {
      return new Response(
        JSON.stringify({
          error: 'No viral videos found for this topic after analysis',
          topic: topic
        }),
        { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // STEP 6: Extract patterns from top videos
    const patterns = extractPatterns(topCandidates)
    console.log('Patterns extracted:', patterns)

    // STEP 7: Generate recommendations using Claude API
    const prompt = buildPrompt(topic, vibe, patterns, topCandidates.slice(0, 5))
    console.log('Calling Claude API...')

    const anthropic = new Anthropic({
      apiKey: Deno.env.get('ANTHROPIC_API_KEY') ?? '',
    })

    const message = await anthropic.messages.create({
      model: 'claude-3-haiku-20240307',
      max_tokens: 1024,
      temperature: 0.8,
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ]
    })

    console.log('Claude response received')

    // Parse recommendations from Claude's response
    const textContent = message.content.find(block => block.type === 'text')
    if (!textContent || textContent.type !== 'text') {
      throw new Error('No text content in Claude response')
    }
    const recommendations = parseRecommendations(textContent.text)

    // STEP 8: Save query to database
    if (userId) {
      await supabaseClient.from('queries').insert({
        user_id: userId,
        topic,
        vibe,
        recommendations,
        matched_candidates: topCandidates.slice(0, 5).map(v => v.video_id)
      })
    }

    // STEP 9: Return comprehensive response
    const processingTime = Date.now() - startTime
    console.log(`✓ Request completed in ${processingTime}ms`)

    return new Response(
      JSON.stringify({
        success: true,
        recommendations,
        topCandidates: topCandidates.slice(0, 3).map(v => ({
          title: v.title,
          thumbnail_url: v.thumbnail_url,
          miles_score: v.miles_score,
          view_count: v.view_count,
          video_id: v.video_id,
          channel_name: v.channel_name
        })),
        meta: {
          videos_analyzed: scrapedVideos.length,
          from_cache: fromCache,
          avg_viral_score: patterns.avgScore,
          processing_time_ms: processingTime,
          topic_searched: topic
        },
        patterns
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Function error:', error)
    return new Response(
      JSON.stringify({
        error: error.message || 'Internal server error',
        details: error.toString()
      }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})

function extractPatterns(videos: any[]) {
  const avgTitleLength = videos.reduce((sum, v) => sum + (v.title?.length || 0), 0) / videos.length
  const avgDuration = videos.reduce((sum, v) => sum + (v.duration_seconds || 30), 0) / videos.length
  const avgScore = videos.reduce((sum, v) => sum + (v.miles_score || 0), 0) / videos.length
  const commonWords = extractCommonWords(videos.map(v => v.title || ''))

  return {
    avgTitleLength: Math.round(avgTitleLength),
    avgDuration: Math.round(avgDuration),
    avgScore: avgScore.toFixed(3),
    commonWords: commonWords.slice(0, 10),
    totalVideos: videos.length
  }
}

function extractCommonWords(titles: string[]): string[] {
  const wordCounts: Record<string, number> = {}
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'be', 'this', 'that', 'it', 'you', 'shorts'
  ])

  titles.forEach(title => {
    title.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .forEach(word => {
        if (word.length > 3 && !stopWords.has(word)) {
          wordCounts[word] = (wordCounts[word] || 0) + 1
        }
      })
  })

  return Object.entries(wordCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([word]) => word)
}

function buildPrompt(topic: string, vibe: string | null, patterns: any, topVideos: any[]): string {
  return `You are a viral content strategist analyzing REAL trending "${topic}" videos that were just scraped from YouTube.

FRESH DATA ANALYSIS (topic-specific):
- ${patterns.totalVideos} "${topic}" videos analyzed
- Average virality score: ${patterns.avgScore} (0-1 scale)
- Average title length: ${patterns.avgTitleLength} characters
- Average duration: ${patterns.avgDuration} seconds
- Common words in ${topic} titles: ${patterns.commonWords.join(', ')}

TOP 5 PERFORMING "${topic}" VIDEOS:
${topVideos.map((v, i) => `${i + 1}. "${v.title}" (viral score: ${(v.miles_score * 100).toFixed(0)}%, ${v.view_count?.toLocaleString() || 'N/A'} views)`).join('\n')}

USER REQUEST:
- Topic: ${topic}
- Desired vibe: ${vibe || 'any style'}

TASK:
Generate 3 specific, actionable video ideas that:
1. Are DIRECTLY related to "${topic}" (not generic)
2. Follow the proven patterns from the real data above
3. Match the ${vibe || 'any'} vibe requested
4. Have viral potential based on what actually works in this niche

OUTPUT FORMAT (JSON only, no markdown):
[
  {
    "title": "Exact video title following the pattern (${patterns.avgTitleLength} chars)",
    "reasoning": "Why this will go viral in the ${topic} niche based on the data (2-3 sentences)"
  },
  {
    "title": "Second video title",
    "reasoning": "Data-driven explanation of viral potential"
  },
  {
    "title": "Third video title",
    "reasoning": "Data-driven explanation of viral potential"
  }
]

IMPORTANT: Return ONLY the JSON array. Use standard double quotes. No markdown formatting.`
}

function parseRecommendations(content: string): any[] {
  try {
    // Try parsing directly
    try {
      const trimmed = content.trim()
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed.map(item => ({
          title: item.title || 'Untitled',
          reasoning: item.reasoning || item.reason || 'No reasoning provided'
        }))
      }
    } catch (e) {
      // Continue to next parsing attempt
    }

    // Extract JSON from code blocks
    const codeBlockMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/)
    if (codeBlockMatch) {
      const parsed = JSON.parse(codeBlockMatch[1])
      if (Array.isArray(parsed)) {
        return parsed.map(item => ({
          title: item.title || 'Untitled',
          reasoning: item.reasoning || item.reason || 'No reasoning provided'
        }))
      }
    }

    // Extract array from content
    const arrayStart = content.indexOf('[')
    const arrayEnd = content.lastIndexOf(']')
    if (arrayStart !== -1 && arrayEnd !== -1 && arrayEnd > arrayStart) {
      let jsonStr = content.substring(arrayStart, arrayEnd + 1)
      jsonStr = jsonStr.replace(/[\u201C\u201D]/g, '"').replace(/[\u2018\u2019]/g, "'")
      const parsed = JSON.parse(jsonStr)
      if (Array.isArray(parsed)) {
        return parsed.map(item => ({
          title: item.title || 'Untitled',
          reasoning: item.reasoning || item.reason || 'No reasoning provided'
        }))
      }
    }

    return [{
      title: 'Parse Error',
      reasoning: 'Failed to parse Claude response. Please try again.'
    }]
  } catch (error) {
    console.error('Parse error:', error)
    return [{
      title: 'Parse Error',
      reasoning: `Failed to parse response: ${error.message}`
    }]
  }
}

/* To invoke locally:
  supabase functions serve scrape-and-recommend --env-file ./supabase/.env.local

  curl -X POST 'http://127.0.0.1:54321/functions/v1/scrape-and-recommend' \
    --header 'Authorization: Bearer <anon-key>' \
    --header 'Content-Type: application/json' \
    --data '{"topic":"skateboarding","vibe":"energetic"}'
*/
