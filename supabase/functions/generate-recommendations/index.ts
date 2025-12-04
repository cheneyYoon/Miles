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

  try {
    const { topic, vibe, userId } = await req.json()
    console.log(`Processing recommendation request: topic="${topic}", vibe="${vibe}"`)

    // Initialize Supabase client
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // 1. Query top-performing candidates with Miles scores
    console.log('Querying top-performing videos...')
    const { data: topCandidates, error: dbError } = await supabaseClient
      .from('candidates')
      .select('*')
      .not('miles_score', 'is', null)  // Only analyzed videos
      .gte('miles_score', 0.5)  // Minimum virality threshold
      .order('miles_score', { ascending: false })
      .limit(20)

    if (dbError) {
      console.error('Database error:', dbError)
      throw dbError
    }

    console.log(`Found ${topCandidates?.length || 0} top candidates`)

    if (!topCandidates || topCandidates.length === 0) {
      return new Response(
        JSON.stringify({
          error: 'No analyzed videos found yet. Please wait for the scraper to run.'
        }),
        { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // 2. Extract patterns from top videos
    const patterns = extractPatterns(topCandidates)
    console.log('Patterns extracted:', patterns)

    // 3. Generate recommendations using Claude API with official SDK
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
    console.log('Usage:', message.usage)

    // Parse recommendations from Claude's response
    const textContent = message.content.find(block => block.type === 'text')
    if (!textContent || textContent.type !== 'text') {
      throw new Error('No text content in Claude response')
    }
    const recommendations = parseRecommendations(textContent.text)

    // 4. Save query to database (if userId provided)
    if (userId) {
      await supabaseClient.from('queries').insert({
        user_id: userId,
        topic,
        vibe,
        recommendations,
        matched_candidates: topCandidates.slice(0, 5).map(v => v.id)
      })
    }

    // 5. Return response
    return new Response(
      JSON.stringify({
        success: true,
        recommendations,
        topCandidates: topCandidates.slice(0, 3).map(v => ({
          title: v.title,
          thumbnail_url: v.thumbnail_url,
          miles_score: v.miles_score,
          view_count: v.view_count,
          video_id: v.video_id
        })),
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
  // Calculate pattern statistics from top videos
  const avgTitleLength = videos.reduce((sum, v) => sum + (v.title?.length || 0), 0) / videos.length
  const avgDuration = videos.reduce((sum, v) => sum + (v.duration_seconds || 30), 0) / videos.length
  const avgScore = videos.reduce((sum, v) => sum + (v.miles_score || 0), 0) / videos.length

  // Extract common words from titles
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
    'of', 'with', 'is', 'are', 'was', 'be', 'this', 'that', 'it', 'you'
  ])

  titles.forEach(title => {
    title.toLowerCase()
      .replace(/[^\w\s]/g, ' ')  // Remove punctuation
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
  return `You are a viral content strategist analyzing short-form video trends on YouTube Shorts and TikTok.

Based on analysis of ${patterns.totalVideos} trending videos, generate 3 specific video ideas for a creator in the "${topic}" niche.

CONTEXT:
- Desired vibe: ${vibe || 'any style'}
- Average virality score: ${patterns.avgScore} (on 0-1 scale)
- Average title length: ${patterns.avgTitleLength} characters
- Average duration: ${patterns.avgDuration} seconds
- Common words in viral titles: ${patterns.commonWords.join(', ')}

TOP PERFORMING VIDEOS IN THIS DATA:
${topVideos.map((v, i) => `${i + 1}. "${v.title}" (virality: ${(v.miles_score * 100).toFixed(0)}%, views: ${v.view_count?.toLocaleString() || 'N/A'})`).join('\n')}

REQUIREMENTS:
1. Each idea should be specific and actionable (not generic advice like "make good content")
2. Include a catchy title that follows the patterns above (similar length and style)
3. Briefly explain WHY this concept has viral potential based on the data (2-3 sentences)
4. Consider the user's topic "${topic}" and vibe "${vibe || 'any'}"
5. Make ideas fresh and unique, not just copies of existing videos

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown, no explanations, just the raw JSON array.
The array must contain exactly 3 objects with this structure:
[
  {
    "title": "Exact video title to use",
    "reasoning": "Why this will go viral based on the data patterns"
  },
  {
    "title": "Second video title",
    "reasoning": "Explanation of viral potential"
  },
  {
    "title": "Third video title",
    "reasoning": "Explanation of viral potential"
  }
]

IMPORTANT:
- Use only standard double quotes ("), not smart quotes
- Escape any quotes within the title or reasoning text
- Return ONLY the JSON array, nothing else

Generate creative, data-driven video ideas now:`
}

function parseRecommendations(content: string): any[] {
  try {
    console.log('Parsing Claude response, length:', content.length)
    console.log('Raw content:', content)

    // First, try parsing the content directly as JSON
    try {
      const trimmed = content.trim()
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed) && parsed.length > 0) {
        console.log('Successfully parsed JSON directly')
        return parsed.map(item => ({
          title: item.title || 'Untitled',
          reasoning: item.reasoning || item.reason || item.explanation || 'No reasoning provided'
        }))
      }
    } catch (e) {
      console.error('Failed to parse full content as JSON:', e)
    }

    // Try to extract JSON from markdown code blocks
    const codeBlockMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/)
    if (codeBlockMatch) {
      try {
        const parsed = JSON.parse(codeBlockMatch[1])
        if (Array.isArray(parsed) && parsed.length > 0) {
          console.log('Successfully parsed from code block')
          return parsed.map(item => ({
            title: item.title || 'Untitled',
            reasoning: item.reasoning || item.reason || item.explanation || 'No reasoning provided'
          }))
        }
      } catch (e) {
        console.error('Failed to parse code block JSON:', e)
      }
    }

    // Try to find and extract just the JSON array
    const arrayStart = content.indexOf('[')
    const arrayEnd = content.lastIndexOf(']')
    if (arrayStart !== -1 && arrayEnd !== -1 && arrayEnd > arrayStart) {
      try {
        let jsonStr = content.substring(arrayStart, arrayEnd + 1)
        console.log('Extracted JSON string (first 500 chars):', jsonStr.substring(0, 500))

        // Try to fix common JSON issues
        // Replace smart quotes with regular quotes
        jsonStr = jsonStr
          .replace(/[\u201C\u201D]/g, '"')  // Smart double quotes
          .replace(/[\u2018\u2019]/g, "'")  // Smart single quotes

        const parsed = JSON.parse(jsonStr)
        if (Array.isArray(parsed) && parsed.length > 0) {
          console.log('Successfully parsed extracted array')
          return parsed.map(item => ({
            title: item.title || 'Untitled',
            reasoning: item.reasoning || item.reason || item.explanation || 'No reasoning provided'
          }))
        }
      } catch (e) {
        console.error('Failed to parse extracted array:', e)
        console.error('Error details:', e.message)
      }
    }

    // Last resort: return error with preview
    console.error('All parsing attempts failed')
    return [{
      title: 'Parse Error - Please Try Again',
      reasoning: 'Claude returned an unexpected format. Content preview: ' + content.substring(0, 200)
    }]
  } catch (error) {
    console.error('Parse error:', error)
    return [{
      title: 'Parse Error',
      reasoning: 'Failed to parse Claude response: ' + error.message
    }]
  }
}

/* To invoke locally:

  1. Start Supabase: `supabase start`
  2. Serve function: `supabase functions serve generate-recommendations --env-file ./supabase/.env.local`
  3. Make an HTTP request:

  curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/generate-recommendations' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
    --header 'Content-Type: application/json' \
    --data '{"topic":"productivity","vibe":"educational"}'

*/
