import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const { title, description, thumbnail_url, view_count, like_count, comment_count, duration_seconds } = await request.json()

    // Validate required fields
    if (!title) {
      return NextResponse.json(
        { error: 'Title is required' },
        { status: 400 }
      )
    }

    // Simulate processing delay (like real ML inference)
    await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 700))

    // Generate mock prediction based on simple heuristics
    const titleLength = title.length
    const hasNumbers = /\d/.test(title)
    const hasQuestion = /\?/.test(title)
    const hasExclamation = /!/.test(title)

    // Base score
    let viralScore = 0.5

    // Adjust based on title characteristics
    if (titleLength > 40 && titleLength < 70) viralScore += 0.1
    if (hasNumbers) viralScore += 0.15
    if (hasQuestion || hasExclamation) viralScore += 0.1

    // Add some randomness
    viralScore += (Math.random() * 0.2) - 0.1

    // Clamp between 0 and 1
    viralScore = Math.max(0, Math.min(1, viralScore))

    // Generate predicted velocity (views per hour)
    const baseVelocity = 500
    const velocityMultiplier = viralScore * 3
    const predictedVelocity = Math.floor(baseVelocity * velocityMultiplier + Math.random() * 500)

    // Calculate confidence (how certain we are about the prediction)
    const confidence = 0.6 + (Math.random() * 0.3)

    return NextResponse.json({
      viral_score: parseFloat(viralScore.toFixed(3)),
      predicted_velocity: predictedVelocity,
      confidence: parseFloat(confidence.toFixed(2)),
      processing_time_ms: Math.floor(300 + Math.random() * 700),
      model_version: 'mock-v1.0',
      note: 'This is a mock endpoint. Real ML inference will be implemented in Phase 3.'
    })

  } catch (error: any) {
    return NextResponse.json(
      { error: 'Failed to process prediction', details: error.message },
      { status: 500 }
    )
  }
}

// Handle OPTIONS for CORS
export async function OPTIONS(request: Request) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
