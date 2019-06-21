from discord.ext import commands
import discord
import asyncio
import json
import Chat
import traceback

startup = True

with open('config.json') as f:
    config = json.load(f)

bot = commands.Bot(command_prefix=config['prefix'], self_bot=False)


# gets chat history
async def get_hist(chan, lim):
    history = []
    hist_itr = chan.history(limit=lim)
    async for m in hist_itr:
        history.insert(0, m.content)
        print(m.content)
    return history

# get response
@bot.command()
async def chat(ctx, *, message):
    async with ctx.channel.typing():
        print('getting response')
        response = Chat.get_response(message)
    await ctx.send(response)

# tell you when the bot is ready
@bot.event
async def on_ready():
    global startup
    if startup:
        startup = False
        print('Logged in')


# main function to start the bot
def main():
    loop = asyncio.get_event_loop()
    loop.create_task(bot.start(config['token']))

    try:
        loop.run_forever()
    finally:
        loop.stop()


if __name__ == '__main__':
    main()
